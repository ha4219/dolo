
import torch as nn
from torch.utils.data import Dataset
from glob import glob
from PIL import Image, ImageFile
import csv
from torch.utils.data.dataset import T_co

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CustomDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        with open(dir_path, 'r') as f:
            rd = csv.reader(f)
            for line in rd:
                self.images.append(line[0])
                self.labels.append(int(line[1]))
        # print(111,self.images, self.labels)

    def __getitem__(self, index) -> T_co:
        img = Image.open(self.images[index])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[index]

    def __len__(self):
        return len(self.images)


class CustomDatasetWithPath(Dataset):
    def __init__(self, dir_path, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        with open(dir_path, 'r') as f:
            rd = csv.reader(f)
            for line in rd:
                self.images.append(line[0])
                self.labels.append(int(line[1]))
        # print(111,self.images, self.labels)

    def __getitem__(self, index) -> T_co:
        img = Image.open(self.images[index])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[index], self.images[index]

    def __len__(self):
        return len(self.images)