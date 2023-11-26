"""Module providing pytorch Dataset for classification."""
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from common import datakeywords as dk

class CustomDataset(Dataset):
    """Class representing pytorch Dataset for classification"""

    def __init__(self, manifest_path, transform=None):
        self.manifest_df = pd.read_csv(manifest_path)
        self.transform = transform

    def __len__(self):
        return len(self.manifest_df)

    def __getitem__(self, idx):
        img_path = self.manifest_df.loc[idx, dk.PATHKEY]
        label = self.manifest_df.loc[idx, dk.LABELKEY]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
