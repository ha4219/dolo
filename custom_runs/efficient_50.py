from custom.create_vision_model import initialize_model
from custom.customPath import _get_path_name, _mkdir_path, _plot_acc, _plot_loss
from custom.dataloader import create_dataloader, create_more_dataloader
from custom.train_model import train_model
import torch
import torch.nn as nn
import torch.cuda

NAME = 'eff_noAni_noDark3'
DEVICE = 'cuda:0'
INPUT_SIZE = 320
BATCH_SIZE = 32
DESC = '100__adam_1e3'  # format: epoch__optimizer_lr_momentum_decay__tuning
EPOCHS = 100
NUM_CLASSES = 5


def main():
    model, _in = initialize_model('efficient', NUM_CLASSES, False, True)
    INPUT_SIZE = _in
    loaders = create_dataloader(_in=INPUT_SIZE, batch_size=BATCH_SIZE, name='noDark20')

    device = DEVICE if torch.cuda.is_available() else 'cpu'

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
