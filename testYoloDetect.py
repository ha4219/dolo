from custom.dataset import CustomDataset
from models.common import DetectMultiBackend
from PIL import Image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from copy import deepcopy

from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator, colors
import cv2
import numpy as np

_in = 640
conf_thres = 0.25  # confidence threshold
iou_thres = 0.45  # NMS IOU threshold
max_det = 1000
classes = 80
agnostic_nms = False

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([_in, _in]),
])

ds = CustomDataset('tmp/erroryoloHighEffNormal.csv', transform=tf)

device = torch.device('cuda:3')
model = DetectMultiBackend(weights='best.pt', device=device, dnn=False, data='data/coco128.yaml', fp16=False)

for i, (im, la) in enumerate(ds):
    im = torch.reshape(im, (1, 3, _in, _in))
    im = im.to(device)
    pred = model(im)
    pred = non_max_suppression(pred, conf_thres, iou_thres, None, agnostic_nms, max_det=max_det)

    names = model.names
    im0 = Image.open(ds.images[i])
    im0 = np.array(im0)
    im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
    annotator = Annotator(im0, line_width=3, example=str(names))

    for j, det in enumerate(pred):
        save_path = f'erroryoloHighEffNormal/{i}.png'

        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()

        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            label = names[c]
            annotator.box_label(xyxy, label, color=colors(c, True))

        im0 = annotator.result()
        cv2.imwrite(save_path, im0)
