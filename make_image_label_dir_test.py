from PIL import Image
from glob import glob

all = glob('/home/oem/lab/jdongha/data/noAni/*/*')

for path in all:
    try:
        img = Image.open(path)
    except:
        print(path)