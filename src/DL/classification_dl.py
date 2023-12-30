"""Module providing pytorch Dataset for classification."""
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from common import datakeywords as dk
from common.utils import get_normalization_transform

def preprocess_image(image: Image.Image, to_predict: bool=False)->torch.Tensor:
    """Preprocess image"""
    data_transform = get_normalization_transform()
    im_tensor = data_transform(image)
    if to_predict:
        im_tensor = im_tensor.unsqueeze(0)
    return im_tensor

class CustomDataset(Dataset):
    """Class representing pytorch Dataset for classification"""

    def __init__(self, manifest_path):
        self.manifest_df = pd.read_csv(manifest_path)

    def __len__(self):
        return len(self.manifest_df)

    def __getitem__(self, idx):
        # Load label
        label = self.manifest_df.loc[idx, dk.LABELKEY]
        # Load image
        img_path = self.manifest_df.loc[idx, dk.PATHKEY]
        image = Image.open(img_path).convert("RGB")
        # Preprocess image
        image = preprocess_image(image)
        return image, label
