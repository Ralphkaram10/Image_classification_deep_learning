import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from common.datakeywords import *

class CustomDataset(Dataset):
    def __init__(self, manifest_path, transform=None):
        self.manifest_df = pd.read_csv(manifest_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.manifest_df)
    
    def __getitem__(self, idx):
        img_path = self.manifest_df.loc[idx, pathkey]
        label = self.manifest_df.loc[idx, labelkey]
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
