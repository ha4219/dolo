import datetime

from custom.create_vision_model import initialize_model
from custom.dataloader import create_dataloader
from custom.dataset import CustomDataset
from custom.layers import FCLayer
from custom.train_model import train_model
from models.yolo import Model
import torch
import torch.nn as nn
from glob import glob

import torch.cuda
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import copy
import time

num_classes = 5
name = "vit"


def main():
    model, _in = initialize_model(name, num_classes, False, True)

    loaders = create_dataloader(_in=_in)

    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    res = train_model(model, loaders, nn.CrossEntropyLoss(), scheduler, optimizer, num_epochs=50,
                      is_inception=False, device=device, label=name)
    now = datetime.datetime.now()
    torch.save(res, f"{now}_{name}.pt")

    return


if __name__ == "__main__":
    main()
