import torch
import torchtext
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import GloVe
import time

start=time.time()
#每篇提取200个单词
TEXT = torchtext.data.Field(lower=True, fix_length=200, batch_first=False)
LABEL = torchtext.data.Field(sequential=False)

train, test = torchtext.datasets.IMDB.splits(TEXT, LABEL)

TEXT.build_vocab(train, max_size=10000, min_freq=10, vectors=None)
LABEL.build_vocab(train)

BATCHSIZE = 256
train_iter, test_iter = torchtext.data.BucketIterator.splits((train, test), batch_size=BATCHSIZE)

embeding_dim = 100
hidden_size = 300


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.em = nn.Embedding(len(TEXT.vocab.stoi), embeding_dim)
        self.lstm = nn.LSTM(embeding_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        bz = x.shape[1]
        h0 = torch.zeros((1, bz, hidden_size)).cuda()
        c0 = torch.zeros((1, bz, hidden_size)).cuda()
        #做词嵌入
        x = self.em(x)
        #然后将词嵌入交给lstm模型处理
        r_o, _ = self.lstm(x, (h0, c0))
        r_o = r_o[-1]
        x = F.relu(self.fc1(r_o))
        x = self.fc2(x)
        return x


model = Net()
if torch.cuda.is_available():
    model.to('cuda')

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


def fit(epoch, model, trainloader, testloader):
    correct = 0
    total = 0
    running_loss = 0

    model.train()
    for b in trainloader:
        x, y = b.text, b.label
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()
    #    exp_lr_scheduler.step()
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct / total

    test_correct = 0
    test_total = 0
    test_running_loss = 0

    model.eval()
    with torch.no_grad():
        for b in testloader:
            x, y = b.text, b.label
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total

    print('epoch: ', epoch,
          'loss： ', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss： ', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
          )

    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc

epochs = 30

train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
                                                                 model,
                                                                 train_iter,
                                                                 test_iter)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

end = time.time()
print(end-start)
