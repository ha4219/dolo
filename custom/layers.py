import torch
import torch.nn as nn


class FlattenLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if not self.training:
            x = x[1]
        x0 = torch.flatten(x[0], 1, -1)
        x1 = torch.flatten(x[1], 1, -1)
        x2 = torch.flatten(x[2], 1, -1)
        x = torch.cat((x0, x1, x2), 1)
        return x


class VGGFCLowLayer(nn.Module):
    def __init__(self, _in=53500, _out=5, _is_flatten=0, _input_size=320, _nc=80):
        super().__init__()
        x2 = _input_size * _input_size
        _in = 3 * (_nc + 5) * (x2 // 64 + x2 // 256 + x2 // 1024) if _is_flatten else _in
        self.layers = nn.ModuleList([
            nn.Linear(_in, 4096),
            nn.Dropout(0.5),
            nn.Linear(4096, _out),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class VGGFCNormalLayer(nn.Module):
    def __init__(self, _in=53500, _out=5, _is_flatten=0, _input_size=320, _nc=80):
        super().__init__()
        x2 = _input_size * _input_size
        _in = 3 * (_nc + 5) * (x2 // 64 + x2 // 256 + x2 // 1024) if _is_flatten else _in
        self.layers = nn.ModuleList([
            nn.Linear(_in, 4096),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.Dropout(0.5),
            nn.Linear(4096, _out),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CustomPoolingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        if not self.training:
            x = x[1]
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]
        x0 = torch.sum(x0, -1)
        x1 = torch.sum(x1, -1)
        x2 = torch.sum(x2, -1)
        x0 = self.avg_pool(self.avg_pool(x0))
        x1 = self.avg_pool(x1)
        # x2 = x2
        x0 = torch.flatten(x0, 1, -1)
        x1 = torch.flatten(x1, 1, -1)
        x2 = torch.flatten(x2, 1, -1)
        x = torch.cat((x0, x1, x2), 1)
        return x


class CustomPoolingOnlyConfidenceLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        if not self.training:
            x = x[1]
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]
        # x0 = torch.sum(x0, -1)
        # x1 = torch.sum(x1, -1)
        # x2 = torch.sum(x2, -1)
        x0 = x0[:, :, :, :, 4]
        x1 = x1[:, :, :, :, 4]
        x2 = x2[:, :, :, :, 4]
        x0 = self.avg_pool(self.avg_pool(x0))
        x1 = self.avg_pool(x1)
        # x2 = x2
        x0 = torch.flatten(x0, 1, -1)
        x1 = torch.flatten(x1, 1, -1)
        x2 = torch.flatten(x2, 1, -1)
        x = torch.cat((x0, x1, x2), 1)
        return x

