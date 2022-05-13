import numpy as np
from torchvision import transforms

from custom.create_vision_model import initialize_model
import torch

from custom.dataset import CustomDatasetWithPath
import matplotlib.pyplot as plt
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

NUM_CLASSES = 5

weights = 'custom_runs/2022-05-07-04:07:07_eff_noAni_noAug_100__adam_1e3/eff_noAni_noAug.pt'
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

ds = CustomDatasetWithPath('/home/oem/lab/jdongha/data/noAni/trainnoDark3.csv', transform=tf)
device = torch.device('cuda:0')
model.to(device)
model.eval()


logs = []
cnts = [0] * 5

for i, (im, la, path) in enumerate(ds):
    im = torch.reshape(im, (1, 3, _in, _in))
    im = im.to(device)
    pred = model(im)
    # t = torch.argmax(pred.cpu()).item()
    _, t = torch.max(pred, 1)
    t = t.cpu().item()
    if t != la:
        cnts[t] += 1
        logs.append((path, la, t))

print(len(ds), len(logs))

d = ['ex-close', 'close', 'full', 'long', 'ex-long']
# plt.bar(['ex-close', 'close', 'full', 'long', 'ex-long'], cnts)
# plt.savefig('tmp/errorEff_noAni.png')

plt.cla()
# for i, (path, truth, predict) in enumerate(logs):
#     img = Image.open(path)
#     plt.imshow(img)
#     plt.title(f'{d[truth]}->{d[predict]}')
#     plt.savefig(f'tmp/{i}.png')

cnts = [[0] * 5 for _ in range(5)]
dd = []

for i in range(5):
    for j in range(5):
        dd.append(f'{i}->{j}')


for path, truth, predict in logs:
    cnts[truth][predict] += 1

plt.imshow(cnts)

for i in range(5):
    for j in range(5):
        text = plt.text(j, i, cnts[i][j],
                       ha="center", va="center", color="w")

plt.savefig('tmp/check.png')

# import csv
#
# with open('tmp/errorEff.csv', 'w') as f:
#     wr = csv.writer(f)
#     for i, log in enumerate(logs):
#         wr.writerow([i, log])
