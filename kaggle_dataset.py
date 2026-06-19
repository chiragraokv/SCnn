import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CIFARKaggle(Dataset):
    def __init__(self, img_dir, labels_csv=None, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        if labels_csv is not None:
            self.data = pd.read_csv(labels_csv)
            self.has_labels = True

            classes = sorted(self.data["label"].unique())
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.has_labels = False

    def __len__(self):
        if self.has_labels:
            return len(self.data)
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_id = self.data.iloc[idx, 0]

        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.has_labels:
            label = self.data.iloc[idx, 1]
            label = self.class_to_idx[label]
            return image, label

        return image

class CIFARTest(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.files = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_name