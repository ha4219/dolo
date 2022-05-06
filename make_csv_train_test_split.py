from glob import glob
import os
import csv
from random import shuffle


def make_csv_train_test_split():
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
        shuffle(tmp)
        tl = len(tmp)
        tll = tl // 10
        train += tmp[:-tll]
        test += tmp[-tll:]

    shuffle(train)
    shuffle(test)

    with open(f'{path}/train.csv', 'w') as f:
        wr = csv.writer(f)
        for name in train:
            wr.writerow([name, name.split('/')[7]])

    with open(f'{path}/test.csv', 'w') as f:
        wr = csv.writer(f)
        for name in test:
            wr.writerow([name, name.split('/')[7]])

if __name__ == '__main__':
    make_csv_train_test_split()