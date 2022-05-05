import torch
import torch.cuda
from torchvision import transforms
from torch.utils.data import DataLoader
import os

from custom.dataset import CustomDataset


def create_dataloader(batch_size=16, workers=8, _in=320):
    trfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([_in, _in]),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

        transforms.ColorJitter(brightness=0.5, hue=0.3),
        transforms.RandomHorizontalFlip(),
    ])

    trfmsv = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([_in, _in]),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

        # transforms.ColorJitter(brightness=0.5, hue=0.3),
        # transforms.RandomHorizontalFlip(),
    ])

    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])

    loaders = {
        "train": DataLoader(CustomDataset("/home/oem/lab/jdongha/data/new/train.csv", trfms),
                            shuffle=True, batch_size=batch_size,
                            num_workers=nw, pin_memory=True),
        "val": DataLoader(CustomDataset("/home/oem/lab/jdongha/data/new/test.csv", trfmsv),
                          shuffle=True, batch_size=batch_size,
                          num_workers=nw, pin_memory=True)}

    return loaders


def create_more_dataloader(batch_size=16, workers=8, _in=320):
    trfms = transforms.Compose([
        transforms.Resize([_in, _in]),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.RandomInvert(),
        transforms.RandomPosterize(bits=2),
        transforms.RandomSolarize(threshold=192.0),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.RandomAutocontrast(),
        transforms.RandomEqualize(),
        transforms.ColorJitter(brightness=0.5, hue=0.3),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),
    ])

    trfmsv = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([_in, _in]),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

        # transforms.ColorJitter(brightness=0.5, hue=0.3),
        # transforms.RandomHorizontalFlip(),
    ])

    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])

    loaders = {
        "train": DataLoader(CustomDataset("/home/oem/lab/jdongha/data/new/train.csv", trfms),
                            shuffle=True, batch_size=batch_size,
                            num_workers=nw, pin_memory=True),
        "val": DataLoader(CustomDataset("/home/oem/lab/jdongha/data/new/test.csv", trfmsv),
                          shuffle=True, batch_size=batch_size,
                          num_workers=nw, pin_memory=True)}

    return loaders
