import torch
import torch.nn as nn
import torch.cuda
import datetime

from custom.customPath import _get_path_name, _mkdir_path, _plot_acc, _plot_loss
from custom.dataloader import create_dataloader
from custom.layers import FlattenLayer, VGGFCLowLayer, EffFCLowLayer
from custom.train_model import train_model
from models.yolo import Model

NAME = 'yoloEFFLow'
DEVICE = 'cuda:2'
INPUT_SIZE = 320
BATCH_SIZE = 16
DESC = '100__adam_1e3'  # format: epoch__optimizer_lr_momentum_decay__tuning
EPOCHS = 100
NUM_CLASSES = 5


def main():
    loaders = create_dataloader(_in=INPUT_SIZE, batch_size=BATCH_SIZE)

    weights = '../best.pt'
    ckpt = torch.load(weights, map_location='cpu')

    csd = ckpt['model'].float().state_dict()
    _model = Model(cfg="../models/yolov5s.yaml")

    _model.load_state_dict(csd, strict=False)

    # feature extracting
    # for param in model.parameters():
    #     param.requires_grad = False

    device = DEVICE if torch.cuda.is_available() else 'cpu'

    model = nn.Sequential(
        _model.to(device),
        EffFCLowLayer(),
    )

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    model, ta, va, tl, vl = train_model(model, loaders, nn.CrossEntropyLoss(),
                                        scheduler, optimizer, num_epochs=EPOCHS,
                                        is_inception=False, device=device, label=NAME)

    path = _get_path_name(NAME, DESC)
    _mkdir_path(path)
    _plot_acc(path, training_acc=ta, validation_acc=va)
    _plot_loss(path, training_loss=tl, validation_loss=vl)
    torch.save(model, f"{path}/{NAME}.pt")
    return


if __name__ == "__main__":
    main()
