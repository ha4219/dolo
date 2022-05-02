from glob import glob
# import csv
#
# dir_path = '/home/oem/lab/jdongha/data/new/test.csv'
# cnt = 0
# with open(dir_path, 'r') as f:
#     rd = csv.reader(f)
#     for line in rd:
#         cnt +=1
#         # images.append(line[0])
#         # labels.append(int(line[1]))
# print(cnt)


# transf
# from torchvision import transforms
# from PIL import Image
# import matplotlib.pyplot as plt
#
#
# trfms = transforms.Compose([
#     # transforms.ToTensor(),
#     # transforms.Resize([_in, _in]),
#     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     # transforms.ColorJitter(brightness=0, hue=0.5),
#     transforms.RandomHorizontalFlip(),
# ])
#
# path = "/home/oem/lab/jdongha/data/origin/1. ex-close(얼굴만)-5743/검은사제들-200/검은사제들.mp4_006010889.png"
# # print(glob("/home/oem/lab/jdongha/data/origin/1. ex-close(얼굴만)-5743/검은사제들-200/*"))
# img = Image.open(path)
# # img.show()
# # Image._show(img)
# # plt.imshow(img)
# # plt.show()
# img = trfms(img)
# print(img)
# plt.imshow(img)
# plt.show()`


import torch
# import gc
# gc.collect()
# torch.cuda.empty_cache()

print(torch.cuda.device_count())