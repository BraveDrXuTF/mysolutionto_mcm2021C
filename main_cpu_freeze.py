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
# feature_path = '/content/drive/MyDrive/garbage_classify/mymodel/vgg16-397923af.pth'
feature_path = '/home/xutengfei/garbage_classify/mymodel/vgg16-397923af.pth'

model = myVGG.MyVGG(1, feature_path=feature_path)
# use BinaryFocalLoss to overcome the lack of positives
loss = focalloss.FocalLoss(alpha=0.95)


for name, params in model.named_parameters():
    # freeze features
    if 'features' in name:
        params.requires_grad = False

optimizer = torch.optim.SGD(filter(
    lambda p: p.requires_grad, model.parameters()), lr=0.001)  # optimizer






batch_size = 7

train_data = MyDataset.MyDataset(transform=myTransform.myTransform)
train_loader = DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)

model.train()  # train model会开放Dropout和BN

sub_batchnum = 0
# use ctrl+C to stop training
while 1:

    subdataset_acc = 0.0

    for i, data in enumerate(train_loader):

        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()  # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = model(inputs)  # 利用 model 的 forward 函数返回预测结果
        batch_loss = loss(train_pred, labels)  # 计算 loss

        batch_loss.backward()  # tensor(item, grad_fn=<NllLossBackward>)
        optimizer.step()  # 以 optimizer 用 gradient 更新参数

        train_pred = train_pred.squeeze()  # train_pred's shape is 20,1

        batch_acc = torch.sum((train_pred > 0.8) == labels)/batch_size

        subdataset_acc = subdataset_acc+batch_acc * \
            batch_size  # here it is a number of right pred
        subdataset_acc_plot = subdataset_acc/((i+1)*batch_size)
        if np.isnan(batch_loss.cpu().item()) == False:
            print('{},{},{},{},{}'.format(
                batch_loss.cpu().item(), batch_acc, subdataset_acc_plot, subdataset_acc, sub_batchnum))

    subdataset_acc = subdataset_acc/train_data.__len__()

    if subdataset_acc > 6/7:   # most samples are right
        # reset dataset ,get new negatives
        sub_batchnum = sub_batchnum+1
        train_data = MyDataset.MyDataset(transform=myTransform.myTransform)
        train_loader = DataLoader(
            dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': subdataset_acc
        }, '/home/xutengfei/garbage_classify/mymodel/mycheckpoint{0}.pth'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())))
