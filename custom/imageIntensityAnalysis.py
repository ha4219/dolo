from PIL import Image
import csv
import matplotlib.pyplot as plt
import numpy as np


path = '/home/oem/lab/jdongha/data/new/test.csv'
MAX = 766
cnts = [0] * MAX

with open(path, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        img = Image.open(row[0])
        img = np.array(img)
        w, h, _ = img.shape
        cnts[int(img.sum() / w / h)] += 1

plt.bar(range(MAX), cnts)
plt.xlabel('intensity')
plt.ylabel('cnt')
plt.savefig('../tmp/originIntensity.png')