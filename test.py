from torchvision import transforms

from custom.create_vision_model import initialize_model
import torch

from custom.dataset import CustomDatasetWithPath
import matplotlib.pyplot as plt

NUM_CLASSES = 5

weights = 'custom_runs/2022-05-05-15:09:07_eff_Aug_100__adam_1e3/eff_Aug.pt'
ckpt = torch.load(weights, map_location='cpu')
csd = ckpt.float().state_dict()
model, _in = initialize_model('efficient', NUM_CLASSES, False, True)
model.load_state_dict(csd, strict=False)


conf_thres = 0.25  # confidence threshold
iou_thres = 0.45  # NMS IOU threshold
max_det = 1000
classes = 80
agnostic_nms = False

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([_in, _in]),
])

ds = CustomDatasetWithPath('/home/oem/lab/jdongha/data/new/test.csv', transform=tf)
device = torch.device('cuda:0')
model.to(device)
model.eval()

logs = []
cnts = [0] * 5

for i, (im, la, path) in enumerate(ds):
    im = torch.reshape(im, (1, 3, _in, _in))
    im = im.to(device)
    pred = model(im)

    if torch.argmax(pred.cpu()).item() != la:
        cnts[torch.argmax(pred.cpu()).item()] += 1
        logs.append(path)

plt.bar(['ex-close', 'close', 'full', 'long', 'ex-long'], cnts)
plt.savefig('tmp/errorEff.png')

import csv

with open('tmp/errorEff.csv', 'w') as f:
    wr = csv.writer(f)
    for i, log in enumerate(logs):
        wr.writerow([i, log])
