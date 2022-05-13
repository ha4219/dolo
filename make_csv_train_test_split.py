from glob import glob
import os
import csv
from random import shuffle
from PIL import Image, ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

TH = 0.05
NAME='noDark5'


def check_pil_th(file, th=0.1):
    try:
        img = Image.open(file)
        img = np.array(img)
        w, h, _ = img.shape
        v = (img.sum() / w / h / 755)
        if v < th:
            return False
        return True
    except:
        print(file)
        return False


def make_csv_train_test_split(th=0.1, pname='noDark5'):
    path = '/home/oem/lab/jdongha/data/noAni'

    labels = [
        '0',
        '1',
        '2',
        '3',
        '4'
    ]

    train = []
    test = []

    for i in range(len(labels)):
        tmp = glob(f'{path}/{i}/*')
        tmp = [item for item in tmp if check_pil_th(item, th)]
        shuffle(tmp)
        tl = len(tmp)
        tll = tl // 10
        train += tmp[:-tll]
        test += tmp[-tll:]

    shuffle(train)
    shuffle(test)

    with open(f'{path}/train{pname}.csv', 'w') as f:
        wr = csv.writer(f)
        for name in train:
            wr.writerow([name, name.split('/')[7]])

    with open(f'{path}/test{pname}.csv', 'w') as f:
        wr = csv.writer(f)
        for name in test:
            wr.writerow([name, name.split('/')[7]])


if __name__ == '__main__':
    make_csv_train_test_split(TH, NAME)
