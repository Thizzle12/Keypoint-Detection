import os

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from keypoint_detection.architecture.efficientnet import EfficientNet
from keypoint_detection.architecture.resnet import ResNet
from keypoint_detection.dataset.data_processor import (
    postprocess_output,
    prepere_for_loss,
)
from keypoint_detection.dataset.keypoint_dataset import KeypointDataset
from keypoint_detection.dataset.data_augmentation import augmentations


def train():
    print("Training model...")
    print("Loading hyperparameters...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.0001
    NUM_WORKERS = 4
    NUM_CLASSES = 30
    CHANNELS = 1
    PIN_MEMORY = True
    USER_PATH = os.path.expanduser("~")
    DATA_PATH = os.path.join(USER_PATH, "Desktop/Datasets/facial keypoints/training/training.csv")

    print("Loading data...")
    aug_fn = augmentations()
    full_dataset = KeypointDataset(DATA_PATH, transforms=aug_fn)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    print("Setting up model...")
    # model = EfficientNet(num_classes=NUM_CLASSES, input_channels=CHANNELS).to(device)
    model = ResNet(num_classes=NUM_CLASSES, input_channels=CHANNELS).to(device)

    # criterion = nn.SmoothL1Loss()

    criterion = nn.MSELoss(reduction="sum")
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    val_loss = np.inf

    print("Training model...")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}: Training...")

        with tqdm(train_loader, unit="Batch") as train_bar:

            for batch_idx, (data, target) in enumerate(train_bar):
                model.train()
                train_bar.set_description(f"Batch {batch_idx}")

                # Convert target to tensor.
                target = prepere_for_loss(target)
                # Adding data and target to device.
                data, target = data.to(device), target.to(device)

                # Zero the gradients.
                optimizer.zero_grad()

                # Forward pass.
                output = model(data)

                # Calculate loss.
                loss = criterion(output, target)

                # Backward pass.
                loss.backward()

                # Update weights.
                optimizer.step()

                train_bar.set_postfix(loss=loss.item())

        with tqdm(val_loader, unit="Batch") as val_bar:
            print("Validating...")

            for batch_idx, (data, target) in enumerate(val_bar):
                with torch.no_grad():
                    model.eval()
                    val_bar.set_description(f"Batch {batch_idx}")

                    # Convert target to tensor.
                    target = prepere_for_loss(target)
                    # Adding data and target to device.
                    data, target = data.to(device), target.to(device)

                    # Zero the gradients.
                    optimizer.zero_grad()

                    # Forward pass.
                    output = model(data)

                    # Calculate loss.
                    loss = criterion(output, target)

                    if loss < val_loss:
                        val_loss = loss
                        torch.save(model.state_dict(), "keypoint_detection/models/best_model.pth")

                    val_bar.set_postfix(loss=loss.item())


if __name__ == "__main__":
    train()
