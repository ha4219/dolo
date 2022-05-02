from custom.dataloader import create_dataloader
from custom.dataset import CustomDataset
from custom.layers import FlattenLayer, VGGFCLowLayer
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
import datetime

num_classes = 5
name = "yolo_FC_Adam"

def main():
    loaders = create_dataloader(_in=320, batch_size=64)

    weights = 'yolov5s.pt'
    ckpt = torch.load(weights, map_location='cpu')

    csd = ckpt['model'].float().state_dict()
    model = Model(cfg="models/yolov5s.yaml")

    model.load_state_dict(csd, strict=False)

    # feature extracting
    for param in model.parameters():
        param.requires_grad = False

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = nn.Sequential(
        model,
        FlattenLayer(),
        VGGFCLowLayer(isFlatten=1, input_size=320),
    )

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    res = train_model(model, loaders, nn.CrossEntropyLoss(), scheduler, optimizer, num_epochs=50,
                      is_inception=False, device=device, label=name)
    now = datetime.datetime.now()
    print(now)
    torch.save(res, f"{now}_{name}.pt")

    return


if __name__ == "__main__":
    main()
