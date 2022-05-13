import matplotlib.pyplot as plt
from glob import glob
import csv

path0 = '/home/oem/lab/jdongha/data/noAni/trainnoDark3.csv'
path1 = '/home/oem/lab/jdongha/data/noAni/testnoDark3.csv'

cnts = [0, 0, 0, 0, 0]

# for i in range(5):
#     tmp = glob(f'{path}/{i}/*')
#     cnts.append(len(tmp))
#
# plt.bar(['ex-close', 'close', 'full', 'long', 'ex-long'], cnts)
# plt.ylabel('cnt')
# plt.title(f'{sum(cnts)}')
# plt.savefig('dataNewBar.png')

with open(path0, 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        cnts[int(line[1])] += 1

with open(path1, 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        cnts[int(line[1])] += 1

plt.bar(['ex-close', 'close', 'full', 'long', 'ex-long'], cnts)
plt.ylabel('cnt')
plt.title(f'{sum(cnts)} th: 0.03')
plt.savefig('datanoDark3Bar.png')