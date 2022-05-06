from glob import glob
from shutil import copy
import os

labels = [
    '1. ex-close(얼굴만)-5743',
    '2. close(상반신)-5082',
    '3. full(전신딱맞게)-5013',
    '4. long(사람키1.5배공간)-5110',
    '5. ex-long(사람키4~5배공간)-5038',
]

name = 'noAni'
base = '/home/oem/lab/jdongha/data'

path = f'{base}/{name}'
os.mkdir(path)

for i in range(len(labels)):
    os.mkdir(f'{path}/{i}')


exc = [
    '코코',
    '애니메트릭스',
    '이세상의한구석에'
]


for i, label in enumerate(labels):
    all = glob(f'/home/oem/lab/jdongha/data/origin/{label}/*/*')

    for j, file in enumerate(all):
        tmp = 0
        for ex in exc:
            if file.split('/')[8].startswith(ex):
                tmp = 1
                break
        if tmp:
            continue
        copy(file, f'{path}/{i}/{j}.png')