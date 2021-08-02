import  h5py
from torchlib import models as nnmodels
import  numpy as np
import random
import os
import cv2
from pytvision.datasets import utility
from CKPDataProvider import *
from Tools import *
from torch.utils.data import random_split
import matplotlib.pyplot as plt

# f = h5py.File('/Users/wangyu/Desktop/利兹上课资料/MSCP/democode/Facial-Expression-Recognition.Pytorch-master/data/CK_data.h5','r')
# print(f.keys())
# print(np.array(f['data_label']))
# print(np.array(f['data_pixel']))
# test = {'t':3}
# print('test', len(test), type(test),test['t'])
# print('len', len(f), type(f))
# print(f['data'])

# from scipy.io import loadmat
# #读取文件
# ucm_seg = loadmat("/Users/wangyu/.datasets/ck/ck.mat")


# for kv in nnmodels.__dict__.items():
#     print(kv)

# classes = ['Neutral - NE', 'Happiness - HA', 'Surprise - SU', 'Sadness - SA', 'Anger - AN', 'Disgust - DI', 'Fear - FR', 'Contempt - CO']
# class_to_idx = {_class: i for i, _class in enumerate(classes)}
# # print(class_to_idx)
#
# toferp = [0, 4, 5, 6, 1, 3, 2, 7 ]
# labels = [0,3,2,1,4,7,6,3]
#
#
# labels = np.array([ toferp[l] for l in labels ])
# print(labels)

# index = np.ones( (10,1) ).astype(int)
# index[0] = 0
# index[8] = 0
# print(index)
# print(index[0])
# indexs = np.where(index == True)[0]
# print(indexs)
# labels = np.array([0,2,4,7,1,5,4,3,2,1])
# labels = labels[ indexs ]
# print(labels)

# a = np.array([[2,3],[4,5]])
# print(a)
# a = a.transpose(1,0)
# # a = a.transpose()
# print(a)

# a = np.arange(16)
# a = a.reshape(2,2,4)
# print(a)
# # b = a.transpose(1,0)
# # print(b)
# print(a[:,0])

# a_shape = a.shape
# print(a_shape[:2:])

# back = np.ones( (640,1024,3), dtype=np.uint8 )*255
# print(back)

# a = random.randint(0,10-1)
# print(a)
# 测试一下render mask部分
# im_h = 5
# im_w = 5
# pad = 2
#
# image = np.arange(75)
# image = image.reshape(3,5,5)
#
# image_pad = np.zeros((3,im_h + 2 * pad, im_w + 2 * pad))
# print(image_pad.shape)
# image_pad[:,pad:-pad, pad:-pad] = image
# image = image_pad
# im_h, im_w = im_h + 2 * pad, im_w + 2 * pad
#
# mask = (image < 1.0).astype(np.uint8)
#
# print(mask)
# dz = 50*random.random()
# print(dz)

# 测试 rn.uniform(-1.0, 1.0)
# a = random.uniform(-1.0, 1.0)
# print(a)

# 测试文件操作方法

ckp_provider = CKPDataProvider()




# ckp_provider.test_hdf5_data()
# r_dir_images = '/Users/wangyu/Downloads/CK数据集/CK+/cohn-kanade-images/'
# image_test = ckp_provider.dict_lab_images[3][0]
# image_path = ckp_provider.get_full_path(image_test)
# print(image_path)
# img = cv2.imread(image_path, 0)
# img = img.transpose(1,0)
# img = ckp_provider.dict_image_imgarray[image_test]
# print(img.shape)
# print(img)
# image = utility.to_channels(img, 3)
# image = img
# print(image.shape)
# print(image)
# cv2.imshow(image_test, img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt.figure("Image") # 图像窗口名称
# plt.imshow(image)
# plt.axis('on') # 关掉坐标轴为 off
# plt.title('image') # 图像题目
# plt.show()






