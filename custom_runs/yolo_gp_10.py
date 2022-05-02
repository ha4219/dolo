from custom.create_vision_model import initialize_model
from custom.dataloader import create_dataloader
from custom.dataset import CustomDataset
from custom.layers import PoolingFCLayer
from custom.train_model import train_model
from models.yolo import Model
import torch
import torch.nn as nn
from glob import glob
import datetime
import torch.cuda
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import copy
import time

num_classes = 5
name = "yolo_gp"


def main():
    # model, _in = initialize_model(name, num_classes, False, True)

    loaders = create_dataloader(_in=320)
    weights = '../yolov5s.pt'
    ckpt = torch.load(weights, map_location='cpu')

    csd = ckpt['model'].float().state_dict()
    model = Model(cfg="../models/yolov5s.yaml")
    model.load_state_dict(csd, strict=False)
    for param in model.parameters():
        param.requires_grad = False
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

    model = nn.Sequential(
        model,
        PoolingFCLayer()
    )
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.1, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    res = train_model(model, loaders, nn.CrossEntropyLoss(),
                      scheduler, optimizer, num_epochs=50,
                      is_inception=False, device=device, label=name)

    now = datetime.datetime.now()
    print(now)
    torch.save(res, f"{now}_{name}.pt")

    return


if __name__ == "__main__":
    main()

