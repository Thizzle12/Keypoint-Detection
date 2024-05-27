import albumentations as A
from albumentations.pytorch import ToTensorV2


def augmentations():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ],
        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=True,
            angle_in_degrees=True,
        ),
    )
