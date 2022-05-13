from PIL import Image
import csv
import matplotlib.pyplot as plt
import numpy as np


# path = '/home/oem/lab/jdongha/data/new/test.csv'
# MAX = 766
# cnts = [0] * MAX
#
# with open(path, 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         img = Image.open(row[0])
#         img = np.array(img)
#         w, h, _ = img.shape
#         cnts[int(img.sum() / w / h)] += 1
#
# plt.bar(range(MAX), cnts)
# plt.xlabel('intensity')
# plt.ylabel('cnt')
# plt.savefig('../tmp/originIntensity.png')
from custom.dataset import CustomDatasetWithPath

ds = CustomDatasetWithPath('/home/oem/lab/jdongha/data/new/test.csv')

res = [0] * 11

for img, label, path in ds:
    # img = Image.open('/home/oem/lab/jdongha/data/noAni/0/3801.png')
    img = np.array(img)
    w, h, _ = img.shape
    v = (img.sum() / w / h/755) * 100
    if v>10:
        continue
    res[round(v)] = (path, v)

print(res)