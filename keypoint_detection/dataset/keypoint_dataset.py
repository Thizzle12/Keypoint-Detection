import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from albumentations import Compose, KeypointParams
from albumentations.pytorch import ToTensorV2


class KeypointDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        img_size: tuple[int, int, int] = (96, 96, 1),
        preprocess: Compose = Compose([ToTensorV2()], keypoint_params=KeypointParams(format="xy")),
        transforms: Compose = None,
    ):
        self.init_data = pd.read_csv(csv_file)
        self.clean_data = self._clean_data()

        self.img_size = img_size

        self.preprocess = preprocess
        self.transforms = transforms

    def _clean_data(self):
        clean_data = self.init_data.dropna(axis=0, how="any", inplace=False)
        clean_data = clean_data.reset_index(drop=True)

        self.data_classes = clean_data.columns
        return clean_data

    def _convert_target(self, target):
        xc = target[0::2]
        yc = target[1::2]

        converted_target = zip(xc, yc)

        return converted_target

    def _recreate_img(self, index):
        x_c = self.clean_data["Image"][index].split(" ")
        x_c = [y for y in x_c]  # create the listed pixels
        clean_imgs_arr = np.array(x_c, dtype="float32")
        clean_imgs_arr = np.reshape(clean_imgs_arr, (96, 96, 1))
        clean_imgs_arr /= 255.0
        return clean_imgs_arr

    def __len__(self):
        return self.clean_data.shape[0]

    def __getitem__(self, index):
        img = self._recreate_img(index)

        target = self.clean_data.iloc[index, :30]
        target = target.to_numpy()
        target = self._convert_target(target)

        if self.transforms:
            transformed = self.transforms(image=img, keypoints=target)
            img = transformed["image"]
            target = transformed["keypoints"]

        preprocessed = self.preprocess(image=img, keypoints=target)
        img = preprocessed["image"].type(torch.float)
        target = preprocessed["keypoints"]

        converted_target = []

        for x, y in target:
            x /= self.img_size[0]
            y /= self.img_size[1]
            converted_target.append((x, y))

        # If image does not contain all keypoints, add (-1, -1) as the missing keypoints.
        if len(converted_target) < 15:
            for _ in range(15 - len(converted_target)):
                converted_target.append([-1.0, -1.0])

        return img, converted_target
