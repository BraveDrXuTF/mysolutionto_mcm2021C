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

model = myVGG.MyVGG(num_classes=1, feature_path=feature_path)
# use BinaryFocalLoss to overcome the lack of positives
loss = focalloss.FocalLoss(alpha=0.95)


for name, params in model.named_parameters():
    # freeze features
    if 'features' in name:
        params.requires_grad = False


# yidingyaofangzaiqianmian
model = model.cuda()
loss = loss.cuda()
optimizer = torch.optim.Adam(filter(
    lambda p: p.requires_grad, model.parameters()), lr=0.001)  # optimizer 使用 Adam


# checkpoint = torch.load('/home/xutengfei/garbage_classify/mymodel/mycheckpoint2021_01_23_20_13_10.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']




num_epoch = 10

train_data = MyDataset.MyDataset(transform=myTransform.myTransform)
print(train_data.__len__())
# print(train_data.printimgs())
train_loader = DataLoader(
    dataset=train_data, batch_size=20, shuffle=True, num_workers=4)

model.train()  # train model会开放Dropout和BN
for epoch in range(num_epoch):
    epoch_start_time = time.time()

    train_acc = 0.0
    train_loss = 0.0

    for i, data in enumerate(train_loader):
        
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        
        train_pred = model(inputs)  # 利用 model 的 forward 函数返回预测结果
        batch_loss = loss(train_pred, labels)  # 计算 loss
        
        optimizer.zero_grad()  # 用 optimizer 將 model 參數的 gradient 歸零
        batch_loss.backward()  # tensor(item, grad_fn=<NllLossBackward>)
        optimizer.step()  # 以 optimizer 用 gradient 更新参数

        train_acc += np.sum(np.argmax(train_pred.data.cpu().numpy(),
                                      axis=1) == labels.cpu().numpy())
        train_loss += batch_loss.cpu().item()
        
        if i%200==0 and i!=0:
            print('train_loss:{0},batch:{1}'.format(batch_loss.cpu().item(), i//200))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, '/home/xutengfei/garbage_classify/mymodel/mycheckpoint{0}.pth'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())))
    
