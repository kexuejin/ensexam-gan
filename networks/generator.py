"""
生成器网络：CoarseNet（粗擦除 + 掩码预测） + RefineNet（精细修复）。

结构对照 EraseNet (STRnet2)，核心改动：
  - 编码器：EraseNet 残差块风格（conv1/conva/convb + ResBlock×8）
  - 跳跃连接：LateralConnection 特征精炼（替代简单 cat）
  - 掩码分支：从 x_mask（瓶颈前 H/16 特征）出发，与 EraseNet 对齐
  - 保留：CBAM 注意力、双 sigmoid 掩码（Ms/Mb）、RefineNet 输入含 Ms
  - RefineNet：EraseNet 风格多尺度空洞卷积，输入保持 cat([Iin, Ms, Ic1])
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.blocks import (DownSample, UpSample, DilatedConvBlock,
                              ResBlock, LateralConnection)


class CoarseNet(nn.Module):
    """粗擦除网络：EraseNet 残差编码器 + 双解码器（修复 + 掩码）。

    Args:
        in_channels:    输入通道数，RGB 固定为 3
        cbam_reduction: CBAM 通道压缩比
    """
    def __init__(self, in_channels: int = 3, cbam_reduction: int = 16):
        super().__init__()
        r = cbam_reduction

        # ── 编码器（EraseNet 风格） ──────────────────────────────────────
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True))   # H/2,  32
        self.conva = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True))   # H/2,  32
        self.convb = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True))   # H/4,  64
        self.res1  = ResBlock(64,  64)
        self.res2  = ResBlock(64,  64)
        self.res3  = ResBlock(64,  128, stride=2)                   # H/8,  128
        self.res4  = ResBlock(128, 128)
        self.res5  = ResBlock(128, 256, stride=2)                   # H/16, 256
        self.res6  = ResBlock(256, 256)
        self.res7  = ResBlock(256, 512, stride=2)                   # H/32, 512
        self.res8  = ResBlock(512, 512)
        self.conv2 = nn.Conv2d(512, 512, 1)                         # H/32, 512（瓶颈）

        # ── 修复解码器（LateralConnection + CBAM UpSample） ─────────────
        self.lat1    = LateralConnection(256)
        self.lat2    = LateralConnection(128)
        self.lat3    = LateralConnection(64)
        self.lat4    = LateralConnection(32)
        self.up1     = UpSample(512, 256, use_cbam=True, reduction=r)
        self.up2     = UpSample(512, 128, use_cbam=True, reduction=r)
        self.up3     = UpSample(256, 64,  use_cbam=True, reduction=r)
        self.up4     = UpSample(128, 32,  use_cbam=True, reduction=r)
        self.up5     = UpSample(64,  32,  use_cbam=True, reduction=r)
        self.out_ic4 = nn.Conv2d(64, 3, 1)
        self.out_ic2 = nn.Conv2d(32, 3, 1)
        self.out_ic1 = nn.Conv2d(32, 3, 3, padding=1)

        # ── 掩码解码器（从瓶颈 d5 出发，保留全局上下文，双 sigmoid 头） ──
        self.mask_up_0   = UpSample(512, 256, use_cbam=True, reduction=r)  # H/32→H/16
        self.mask_up_a   = UpSample(512, 256, use_cbam=True, reduction=r)
        self.mask_conv_a = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.mask_up_b   = UpSample(256, 128, use_cbam=True, reduction=r)
        self.mask_conv_b = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.mask_up_c   = UpSample(128, 64, use_cbam=True, reduction=r)
        self.mask_conv_c = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.mask_up_d   = UpSample(64, 32, use_cbam=True, reduction=r)
        self.out_ms      = nn.Conv2d(32, 1, 1)                      # 软笔画掩码
        self.out_mb      = nn.Conv2d(32, 1, 1)                      # 文本块掩码

    def forward(self, x: torch.Tensor):
        # 编码
        x      = self.conv1(x)
        x      = self.conva(x)
        con_x1 = x                                   # H/2,  32
        x      = self.convb(x)
        x      = self.res1(x)
        con_x2 = x                                   # H/4,  64
        x      = self.res2(x)
        x      = self.res3(x)
        con_x3 = x                                   # H/8,  128
        x      = self.res4(x)
        x      = self.res5(x)
        con_x4 = x                                   # H/16, 256
        x      = self.res6(x)
        x      = self.res7(x)
        x      = self.res8(x)
        d5     = self.conv2(x)                       # H/32, 512（瓶颈）

        # 修复解码
        u1  = torch.cat([self.lat1(con_x4), self.up1(d5)],  dim=1)  # H/16, 512
        u2  = torch.cat([self.lat2(con_x3), self.up2(u1)],  dim=1)  # H/8,  256
        xo1 = self.up3(u2)                                           # H/4,  64
        Ic4 = torch.tanh(self.out_ic4(xo1))
        u3  = torch.cat([self.lat3(con_x2), xo1], dim=1)            # H/4,  128
        xo2 = self.up4(u3)                                           # H/2,  32
        Ic2 = torch.tanh(self.out_ic2(xo2))
        u4  = torch.cat([self.lat4(con_x1), xo2], dim=1)            # H/2,  64
        Ic1 = torch.tanh(self.out_ic1(self.up5(u4)))                 # H,    3

        # 掩码解码（从瓶颈 d5 出发，保留全局上下文）
        mm = self.mask_up_a(torch.cat([self.mask_up_0(d5), con_x4], dim=1))  # H/8,  256
        mm = self.mask_conv_a(mm)                                     # H/8,  128
        mm = self.mask_up_b(torch.cat([mm, con_x3], dim=1))         # H/4,  128
        mm = self.mask_conv_b(mm)                                     # H/4,  64
        mm = self.mask_up_c(torch.cat([mm, con_x2], dim=1))         # H/2,  64
        mm = self.mask_conv_c(mm)                                     # H/2,  32
        mm = self.mask_up_d(torch.cat([mm, con_x1], dim=1))         # H,    32
        Ms = torch.sigmoid(self.out_ms(mm))
        Mb = torch.sigmoid(self.out_mb(mm))

        return Ms, Mb, Ic4, Ic2, Ic1


class RefineNet(nn.Module):
    """精细修复网络：EraseNet 风格多尺度空洞卷积，感受野更大。

    Args:
        in_channels: 输入通道数 = Iin(3) + Ms(1) + Ic1(3) = 7
    """
    def __init__(self, in_channels: int = 7):
        super().__init__()
        cnum = 32

        # 编码
        self.conva = nn.Sequential(
            nn.Conv2d(in_channels, cnum, 5, padding=2, bias=False),
            nn.BatchNorm2d(cnum), nn.ReLU(inplace=True))
        self.down1 = DownSample(cnum,     cnum * 2)                  # H/2, 64
        self.convc = nn.Sequential(
            nn.Conv2d(cnum * 2, cnum * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(cnum * 2), nn.ReLU(inplace=True))
        self.down2 = DownSample(cnum * 2, cnum * 4)                  # H/4, 128
        self.conve = nn.Sequential(
            nn.Conv2d(cnum * 4, cnum * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(cnum * 4), nn.ReLU(inplace=True))
        self.convf = nn.Sequential(
            nn.Conv2d(cnum * 4, cnum * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(cnum * 4), nn.ReLU(inplace=True))

        # 多尺度空洞卷积（参照 EraseNet astrous_net）
        self.astrous = nn.Sequential(
            DilatedConvBlock(cnum * 4, cnum * 4, dilation=2),
            DilatedConvBlock(cnum * 4, cnum * 4, dilation=4),
            DilatedConvBlock(cnum * 4, cnum * 4, dilation=8),
            DilatedConvBlock(cnum * 4, cnum * 4, dilation=16),
        )
        self.convk = nn.Sequential(
            nn.Conv2d(cnum * 4, cnum * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(cnum * 4), nn.ReLU(inplace=True))
        self.convl = nn.Sequential(
            nn.Conv2d(cnum * 4, cnum * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(cnum * 4), nn.ReLU(inplace=True))

        # 解码（cat 自身跳跃连接）
        self.up1   = UpSample(cnum * 8, cnum * 2)                    # cat([x,x_c2]) H/4→H/2
        self.convm = nn.Sequential(
            nn.Conv2d(cnum * 2, cnum * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(cnum * 2), nn.ReLU(inplace=True))
        self.up2   = UpSample(cnum * 4, cnum)                        # cat([x,x_c1]) H/2→H
        self.convn = nn.Sequential(
            nn.Conv2d(cnum, cnum // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(cnum // 2), nn.ReLU(inplace=True),
            nn.Conv2d(cnum // 2, 3, 3, padding=1))

    def forward(
        self,
        x: torch.Tensor,
        return_reconstruction_feature: bool = False,
    ):
        x    = self.conva(x)
        x    = self.down1(x)
        x    = self.convc(x)
        x_c1 = x                                      # H/2, 64
        x    = self.down2(x)
        x    = self.conve(x)
        x    = self.convf(x)
        x_c2 = x                                      # H/4, 128
        x    = self.astrous(x)
        x    = self.convk(x)
        x    = self.convl(x)
        x    = self.up1(torch.cat([x, x_c2], dim=1)) # H/2, 64
        x    = self.convm(x)
        x    = self.up2(torch.cat([x, x_c1], dim=1)) # H,   32
        reconstruction_feature = self.convn[:-1](x)
        Ire = torch.tanh(self.convn[-1](reconstruction_feature))
        if return_reconstruction_feature:
            return Ire, reconstruction_feature
        return Ire


class UniversalResidualAdapterSidecar(nn.Module):
    """Continuous residual-only sidecar mixed from reconstruction features."""

    def __init__(
        self,
        feature_channels: int = 16,
        output_channels: int = 3,
        adapter_count: int = 3,
        hidden_channels: int = 16,
        residual_bound: float = 12.0 / 255.0,
        fallback_residual_abs_max: float | None = None,
        residual_parameterization: str = "free_rgb",
    ):
        super().__init__()
        if adapter_count != 3:
            raise ValueError("adapter_count must be 3")
        if residual_bound <= 0.0 or residual_bound > 12.0 / 255.0:
            raise ValueError("residual_bound must be in (0, 12/255]")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if residual_parameterization not in {"free_rgb", "primary_edit_direction"}:
            raise ValueError(
                "residual_parameterization must be free_rgb or "
                "primary_edit_direction"
            )
        self.adapter_count = int(adapter_count)
        self.residual_bound = float(residual_bound)
        self.residual_parameterization = residual_parameterization
        self.fallback_residual_abs_max = (
            float(fallback_residual_abs_max)
            if fallback_residual_abs_max is not None
            else float(residual_bound) * 4.0
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, self.adapter_count),
        )
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(feature_channels, hidden_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        hidden_channels,
                        1 if residual_parameterization == "primary_edit_direction"
                        else output_channels,
                        1,
                    ),
                )
                for _ in range(self.adapter_count)
            ]
        )
        self.initial_global_residual_scale = 1e-3
        self.global_residual_scale = nn.Parameter(
            torch.tensor(self.initial_global_residual_scale, dtype=torch.float32)
        )
        self.reset_residual_to_zero()

    def reset_residual_to_zero(self) -> None:
        for adapter in self.adapters:
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)
        nn.init.constant_(self.global_residual_scale, self.initial_global_residual_scale)

    def forward(
        self,
        reconstruction_feature: torch.Tensor,
        baseline_output: torch.Tensor,
        input_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        gate_logits = self.gate(reconstruction_feature)
        gate_weights = torch.softmax(gate_logits, dim=1)
        adapter_residuals = torch.stack(
            [adapter(reconstruction_feature) for adapter in self.adapters],
            dim=1,
        )
        mixed_residual = (
            gate_weights.view(gate_weights.shape[0], self.adapter_count, 1, 1, 1)
            * adapter_residuals
        ).sum(dim=1)
        if self.residual_parameterization == "primary_edit_direction":
            if input_image is None or input_image.shape != baseline_output.shape:
                raise ValueError(
                    "primary_edit_direction requires input_image matching "
                    "baseline_output"
                )
            nonnegative_magnitude = torch.where(
                mixed_residual >= 0,
                mixed_residual,
                torch.zeros_like(mixed_residual),
            )
            bounded_magnitude = self.residual_bound * torch.tanh(
                nonnegative_magnitude
            )
            nonnegative_scale = torch.where(
                self.global_residual_scale >= 0,
                self.global_residual_scale,
                torch.zeros_like(self.global_residual_scale),
            )
            applied_scale = torch.tanh(nonnegative_scale)
            primary_edit = baseline_output - input_image
            edit_norm = primary_edit.abs().amax(dim=1, keepdim=True)
            primary_direction = torch.where(
                edit_norm > 0,
                primary_edit / edit_norm.clamp_min(1e-12),
                torch.zeros_like(primary_edit),
            )
            scaled_residual = (
                applied_scale
                * bounded_magnitude
                * primary_direction
            )
        else:
            bounded_residual = self.residual_bound * torch.tanh(mixed_residual)
            applied_scale = torch.tanh(self.global_residual_scale)
            scaled_residual = (
                applied_scale * bounded_residual
            )
        fallback_code = baseline_output.new_tensor(0.0)
        invalid = (
            not torch.isfinite(gate_weights).all()
            or not torch.isfinite(scaled_residual).all()
            or scaled_residual.shape != baseline_output.shape
            or float(scaled_residual.detach().abs().max().cpu())
            > self.fallback_residual_abs_max
        )
        if invalid:
            candidate = baseline_output
            fallback_code = baseline_output.new_tensor(1.0)
            residual_for_telemetry = torch.zeros_like(baseline_output)
        else:
            residual_for_telemetry = scaled_residual
            if self.training:
                candidate = torch.clamp(baseline_output + scaled_residual, -1.0, 1.0)
            else:
                residual_is_zero = bool((scaled_residual.detach() == 0).all().cpu())
                candidate = (
                    baseline_output
                    if residual_is_zero
                    else torch.clamp(baseline_output + scaled_residual, -1.0, 1.0)
                )
        gate_entropy = -(gate_weights * gate_weights.clamp_min(1e-12).log()).sum(dim=1)
        telemetry = {
            "ura_gate_entropy_mean": gate_entropy.mean(),
            "ura_gate_max_mean": gate_weights.max(dim=1).values.mean(),
            "ura_residual_abs_mean": residual_for_telemetry.detach().abs().mean(),
            "ura_residual_abs_max": residual_for_telemetry.detach().abs().amax(),
            "ura_residual_scale_abs": applied_scale.detach().abs(),
            "ura_fallback_code": fallback_code,
        }
        return candidate, telemetry


class Generator(nn.Module):
    """完整生成器：CoarseNet → RefineNet → 融合输出。

    Args:
        cfg: model 子配置字典，含 coarse_in_channels / refine_in_channels / cbam_reduction
    """
    def __init__(self, cfg: dict = None):
        super().__init__()
        if cfg is None:
            cfg = {'coarse_in_channels': 3, 'refine_in_channels': 7, 'cbam_reduction': 16}
        self.coarse = CoarseNet(in_channels=cfg['coarse_in_channels'],
                                cbam_reduction=cfg['cbam_reduction'])
        self.refine = RefineNet(in_channels=cfg['refine_in_channels'])
        sidecar_cfg = cfg.get("universal_residual_adapter_sidecar", {}) or {}
        if isinstance(sidecar_cfg, bool):
            sidecar_cfg = {"enabled": sidecar_cfg}
        if not isinstance(sidecar_cfg, dict):
            raise ValueError("universal_residual_adapter_sidecar must be a mapping")
        self.universal_residual_adapter_sidecar_enabled = bool(
            sidecar_cfg.get("enabled", False)
        )
        if self.universal_residual_adapter_sidecar_enabled:
            self.universal_residual_adapter_sidecar = UniversalResidualAdapterSidecar(
                feature_channels=int(sidecar_cfg.get("feature_channels", 16)),
                output_channels=3,
                adapter_count=int(sidecar_cfg.get("adapter_count", 3)),
                hidden_channels=int(sidecar_cfg.get("hidden_channels", 16)),
                residual_bound=float(sidecar_cfg.get("residual_bound", 12.0 / 255.0)),
                fallback_residual_abs_max=sidecar_cfg.get("fallback_residual_abs_max"),
                residual_parameterization=str(
                    sidecar_cfg.get("residual_parameterization", "free_rgb")
                ),
            )

    def forward(
        self,
        Iin: torch.Tensor,
        return_reconstruction_feature: bool = False,
        return_universal_sidecar_telemetry: bool = False,
    ):
        Ms, Mb, Ic4, Ic2, Ic1 = self.coarse(Iin)
        need_reconstruction_feature = (
            return_reconstruction_feature
            or self.universal_residual_adapter_sidecar_enabled
        )
        refine_input = torch.cat([Iin, Ms, Ic1], dim=1)
        reconstruction_feature = None
        if need_reconstruction_feature:
            Ire, reconstruction_feature = self.refine(
                refine_input,
                return_reconstruction_feature=True,
            )
        else:
            Ire = self.refine(refine_input)
        Icomp = Ire * Mb + Iin * (1 - Mb)
        universal_sidecar_telemetry = None
        if self.universal_residual_adapter_sidecar_enabled:
            Icomp, universal_sidecar_telemetry = self.universal_residual_adapter_sidecar(
                reconstruction_feature,
                Icomp,
                Iin,
            )
        elif return_universal_sidecar_telemetry:
            universal_sidecar_telemetry = {
                "ura_enabled": Icomp.new_tensor(0.0),
                "ura_fallback_code": Icomp.new_tensor(0.0),
            }
        outputs = (Ms, Mb, Ic4, Ic2, Ic1, Ire, Icomp)
        if return_reconstruction_feature and return_universal_sidecar_telemetry:
            return outputs, reconstruction_feature, universal_sidecar_telemetry
        if return_reconstruction_feature:
            return outputs, reconstruction_feature
        if return_universal_sidecar_telemetry:
            return outputs, universal_sidecar_telemetry
        return outputs
