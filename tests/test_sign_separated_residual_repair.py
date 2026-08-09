import inspect
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.analysis.audit_sign_separated_residual_repair import (
    compute_synthetic_loss_terms,
    run_audit,
)
from scripts.infer.patch_cleanup_erasemap import (
    EraseMapCleanupNet,
    ResidualDeltaCleanupNet,
    SignSeparatedResidualDeltaCleanupNet,
    build_model,
    infer_full_page,
    load_model,
)


MODEL_TYPE = "sign_separated_residual_delta"
BOUND = 0.08


def final_bias_gradient(model, head_name: str) -> float:
    head = getattr(model, head_name)
    gradient = head[-1].bias.grad
    return 0.0 if gradient is None else float(gradient.detach().abs().sum())


class SignSeparatedResidualRepairTest(unittest.TestCase):
    def make_model(self) -> SignSeparatedResidualDeltaCleanupNet:
        torch.manual_seed(20260809)
        return SignSeparatedResidualDeltaCleanupNet(
            residual_delta_bound=BOUND
        )

    def test_legacy_cleanup_models_keep_shape_and_public_surface(self) -> None:
        expected = {
            "erasemap": (EraseMapCleanupNet, 384612),
            "residual_delta": (ResidualDeltaCleanupNet, 384612),
        }
        x = torch.full((1, 3, 16, 16), 0.5)
        for model_type, (model_class, parameter_count) in expected.items():
            with self.subTest(model_type=model_type):
                kwargs = {"residual_delta_scale": 0.08}
                model = build_model(model_type, **kwargs).eval()
                self.assertIsInstance(model, model_class)
                self.assertEqual(sum(p.numel() for p in model.parameters()), parameter_count)
                self.assertEqual(len(model.state_dict()), 32)
                with torch.no_grad():
                    outputs = model(x)
                self.assertEqual(
                    [tuple(value.shape) for value in outputs],
                    [(1, 3, 16, 16), (1, 1, 16, 16), (1, 3, 16, 16)],
                )

        self.assertEqual(
            set(inspect.signature(infer_full_page).parameters),
            {
                "model",
                "image",
                "device",
                "tile_size",
                "stride",
                "alpha_threshold",
            },
        )

    def test_zero_initialization_is_exact_identity_without_global_scale(self) -> None:
        model = self.make_model().eval()
        x = torch.rand(2, 3, 16, 16) * 0.8 + 0.1
        with torch.no_grad():
            pred, edit_alpha, clean_candidate = model(x)
            components = model.forward_components(x)

        self.assertTrue(torch.equal(pred, x))
        self.assertTrue(torch.equal(clean_candidate, x))
        self.assertTrue(torch.equal(components["signed_delta"], torch.zeros_like(x)))
        self.assertTrue(torch.allclose(edit_alpha, torch.full_like(edit_alpha, 2.0 / 3.0)))
        self.assertFalse(
            any(
                "global" in name and "scale" in name
                for name, _ in model.named_parameters()
            )
        )
        self.assertTrue(
            torch.equal(
                model.bright_magnitude_head[-1].weight,
                torch.zeros_like(model.bright_magnitude_head[-1].weight),
            )
        )
        self.assertTrue(
            torch.equal(
                model.dark_magnitude_head[-1].weight,
                torch.zeros_like(model.dark_magnitude_head[-1].weight),
            )
        )
        self.assertEqual(set(inspect.signature(model.forward).parameters), {"x"})

    def test_forced_routes_have_correct_sign_and_shared_bound(self) -> None:
        x = torch.full((1, 3, 8, 8), 0.5)
        for route_index, expected_sign in ((1, 1), (2, -1), (0, 0)):
            with self.subTest(route_index=route_index):
                model = self.make_model().eval()
                with torch.no_grad():
                    model.route_head[-1].weight.zero_()
                    model.route_head[-1].bias.fill_(-100.0)
                    model.route_head[-1].bias[route_index] = 100.0
                    model.bright_magnitude_head[-1].bias.fill_(-100.0)
                    model.dark_magnitude_head[-1].bias.fill_(100.0)
                    pred, _, _ = model(x)
                delta = pred - x
                self.assertLessEqual(float(delta.abs().max()), BOUND + 1e-7)
                if expected_sign > 0:
                    self.assertTrue(torch.all(delta >= 0.0))
                    self.assertGreater(float(delta.max()), 0.0)
                elif expected_sign < 0:
                    self.assertTrue(torch.all(delta <= 0.0))
                    self.assertLess(float(delta.min()), 0.0)
                else:
                    self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))

    def test_matching_branch_gradients_live_and_opposite_branch_isolated(self) -> None:
        for direction in (1, -1):
            with self.subTest(direction=direction):
                model = self.make_model().train()
                x = torch.full((1, 3, 8, 8), 0.5)
                target = x + direction * 0.05
                terms = compute_synthetic_loss_terms(model, x, target)
                terms["loss"].backward()

                matching = (
                    "bright_magnitude_head"
                    if direction > 0
                    else "dark_magnitude_head"
                )
                opposite = (
                    "dark_magnitude_head"
                    if direction > 0
                    else "bright_magnitude_head"
                )
                self.assertGreater(final_bias_gradient(model, matching), 0.0)
                self.assertEqual(final_bias_gradient(model, opposite), 0.0)
                self.assertGreater(
                    float(model.route_head[-1].bias.grad.detach().abs().sum()),
                    0.0,
                )

    def test_two_step_update_produces_only_the_target_direction(self) -> None:
        for direction in (1, -1):
            with self.subTest(direction=direction):
                model = self.make_model().train()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                x = torch.full((1, 3, 8, 8), 0.5)
                target = x + direction * 0.05
                for _ in range(2):
                    optimizer.zero_grad()
                    terms = compute_synthetic_loss_terms(model, x, target)
                    terms["loss"].backward()
                    optimizer.step()

                with torch.no_grad():
                    pred, _, _ = model(x)
                delta = pred - x
                self.assertGreater(float(delta.abs().max()), 0.0)
                self.assertLessEqual(float(delta.abs().max()), BOUND + 1e-7)
                self.assertEqual(int((delta * direction < -1e-8).sum()), 0)

    def test_identity_target_keeps_both_magnitudes_zero(self) -> None:
        model = self.make_model().train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        x = torch.full((1, 3, 8, 8), 0.5)
        for _ in range(2):
            optimizer.zero_grad()
            terms = compute_synthetic_loss_terms(model, x, x.clone())
            terms["loss"].backward()
            optimizer.step()
        with torch.no_grad():
            pred, _, _ = model(x)
            components = model.forward_components(x)
        self.assertTrue(torch.equal(pred, x))
        self.assertTrue(
            torch.equal(
                components["bright_magnitude"],
                torch.zeros_like(components["bright_magnitude"]),
            )
        )
        self.assertTrue(
            torch.equal(
                components["dark_magnitude"],
                torch.zeros_like(components["dark_magnitude"]),
            )
        )

    def test_new_model_checkpoint_roundtrip_is_exact(self) -> None:
        model = self.make_model().eval()
        x = torch.rand(1, 3, 16, 16) * 0.8 + 0.1
        with torch.no_grad():
            model.route_head[-1].bias.copy_(torch.tensor([0.0, 1.0, -1.0]))
            model.bright_magnitude_head[-1].bias.fill_(0.25)
            expected = model(x)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            torch.save(
                {
                    "args": {
                        "model_type": MODEL_TYPE,
                        "residual_delta_bound": BOUND,
                    },
                    "model": model.state_dict(),
                },
                path,
            )
            restored = load_model(path, torch.device("cpu"))
            with torch.no_grad():
                actual = restored(x)[0]

        self.assertIsInstance(restored, SignSeparatedResidualDeltaCleanupNet)
        self.assertTrue(torch.equal(actual, expected))

    def test_legacy_checkpoint_roundtrip_is_exact(self) -> None:
        x = torch.full((1, 3, 16, 16), 0.5)
        for model_type in ("erasemap", "residual_delta"):
            with self.subTest(model_type=model_type):
                torch.manual_seed(19)
                model = build_model(model_type, residual_delta_scale=0.08).eval()
                with torch.no_grad():
                    expected = model(x)
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = Path(tmpdir) / "legacy.pt"
                    torch.save(
                        {
                            "args": {
                                "model_type": model_type,
                                "residual_delta_scale": 0.08,
                            },
                            "model": model.state_dict(),
                        },
                        path,
                    )
                    restored = load_model(path, torch.device("cpu"))
                    with torch.no_grad():
                        actual = restored(x)
                self.assertTrue(
                    all(
                        torch.equal(expected_value, actual_value)
                        for expected_value, actual_value in zip(expected, actual)
                    )
                )

    def test_fail_closed_audit_passes(self) -> None:
        result = run_audit()
        self.assertEqual(result["terminal"], "PASS")
        self.assertTrue(result["exact_identity_init"])
        self.assertTrue(result["zero_magnitude_projection_init"])
        self.assertFalse(result["training_cli_enabled"])
        self.assertEqual(result["opposed_pixel_count"], 0)
        self.assertGreater(result["serialization_delta_abs_max"], 0.0)
        self.assertEqual(
            len(result["forced_route_cases"][0]["route_probability_mean"]),
            3,
        )


if __name__ == "__main__":
    unittest.main()
