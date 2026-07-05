"""
基于 albumentations 的 Paired 数据增强。
albumentations 的 additional_targets 机制保证 Iin 和 Igt 施加完全相同的随机变换。
所有概率和强度参数均从 config.yaml 的 data.augmentation 节读取。
"""
import albumentations as A


def _variance_range_to_std_range(var_range):
    """Convert legacy 0-255 variance config to albumentations 2.x std_range."""
    lo, hi = var_range
    # Albumentations 2.x expects std normalized to [0, 1].
    return (float(lo) ** 0.5 / 255.0, float(hi) ** 0.5 / 255.0)


DEFAULT_AUGMENTATION = {
    'horizontal_flip_p': 0.5,
    'vertical_flip_p': 0.3,
    'rotate90_p': 0.3,
    'brightness_limit': 0.2,
    'contrast_limit': 0.2,
    'brightness_contrast_p': 0.5,
    'gauss_noise_var_limit': [5.0, 20.0],
    'gauss_noise_p': 0.3,
}


def get_train_augmentation(aug_cfg: dict) -> A.Compose:
    """
    构建训练增强流水线。

    Args:
        aug_cfg: config.yaml 中 data.augmentation 子字典

    Returns:
        A.Compose 实例，调用方式：
            result = aug(image=Iin, gt=Igt)
            Iin_aug, Igt_aug = result['image'], result['gt']
    """
    cfg = {**DEFAULT_AUGMENTATION, **(aug_cfg or {})}

    return A.Compose([
        A.HorizontalFlip(
            p=cfg['horizontal_flip_p'],
        ),
        A.VerticalFlip(
            p=cfg['vertical_flip_p'],
        ),
        A.RandomRotate90(
            p=cfg['rotate90_p'],
        ),
        A.RandomBrightnessContrast(
            brightness_limit=cfg['brightness_limit'],
            contrast_limit=cfg['contrast_limit'],
            p=cfg['brightness_contrast_p'],
        ),
        A.GaussNoise(
            std_range=_variance_range_to_std_range(cfg['gauss_noise_var_limit']),
            p=cfg['gauss_noise_p'],
        ),
    ], additional_targets={
        'gt': 'image',   # gt 与 image 施加完全相同的随机变换
        'mb': 'mask',    # Mb 掩码与 image 施加相同的空间变换（亮度/噪声不影响 mask）
        'box_preserve': 'mask',
    })
