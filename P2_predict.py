import torchvision
import myVGG
import MyDataset
import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time
from PIL import Image
from torch.autograd import Variable
import focalloss
import myTransform
feature_path = '/home/xutengfei/garbage_classify/mymodel/vgg16-397923af.pth'
model = myVGG.MyVGG(1, feature_path=feature_path)
checkpoint = torch.load('/home/xutengfei/garbage_classify/mymodel/mcm_final1.pth',map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
acc=checkpoint['acc']
# print(acc)
model.eval()
train_data = MyDataset.MyDataset(transform=myTransform.myTransform,negative_num=1e6)
train_loader = DataLoader(
    dataset=train_data, batch_size=10, shuffle=True, num_workers=4)
acc=0
for i, data in enumerate(train_loader):
    inputs, labels = data
    train_pred = model(inputs)
    train_pred = train_pred.squeeze()
    acc = torch.sum((train_pred > 0.8) == labels)+acc
    print(acc)

acc /= float(train_data.__len__())
print(acc)

