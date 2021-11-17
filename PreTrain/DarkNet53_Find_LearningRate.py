#---------------step0:Common Definition-----------------
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

batch_size = 16
img_size = 256
min_lr = 1e-4
max_lr = 1
mul_factor = 1.2
weight_decay = 5e-4
momentum = 0.9
min_val_loss = 9999999999

def accuracy(output, target, topk=(1, 5)):

    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k / batch_size)
    return res

#---------------step1:Dataset-------------------
from ImageNet_DataSet import ImageNetMini
dataSet = ImageNetMini(dataSetDir="../DataSet/imagenet-mini/train",classesFilePath="../DataSet/imagenet-mini/classDict.pth", img_size=256)

#---------------step2:Model-------------------
from DarkNet53 import DarkNet53
darkNet53 = DarkNet53(class_num=1000).to(device=device)
darkNet53.weight_init()

#---------------step3:LossFunction-------------------
loss_function = nn.CrossEntropyLoss().to(device=device)

#---------------step4:Optimizer-------------------
import torch.optim as optim
#optimizer_Adam = optim.Adam(darkNet53.parameters(),lr=lr,weight_decay=weight_decay)
#使用余弦退火动态调整学习率
#lr_reduce_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_Adam , T_max=20, eta_min=1e-4, last_epoch=-1)
#lr_reduce_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer_Adam, T_0=2, T_mult=2)

#---------------step5:Train-------------------
from tensorboardX import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

train_loader = DataLoader(dataSet, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
darkNet53.train()

lr_s = []
loss_s = []
max_grad = 0
max_grad_lr = 0

for batch_index, batch_train in enumerate(train_loader):

    optimizer_SGD = optim.SGD(darkNet53.parameters(), lr=min_lr)

    print(optimizer_SGD)

    train_data = batch_train[0].float().to(device=device)
    label_data = batch_train[1].to(device=device)
    net_out = darkNet53(train_data)
    loss = loss_function(net_out,label_data) / batch_size

    loss.backward()
    optimizer_SGD.step()
    optimizer_SGD.zero_grad()
    batch_loss = loss.item() * batch_size

    lr_s.append(min_lr)
    loss_s.append(batch_loss)

    if min_lr >= max_lr:
        break

    if len(loss_s) > 1:
        grad = (loss_s[-2] - loss_s[-1]) / (lr_s[-2] - lr_s[-1])
        if max_grad < grad:
            max_grad = grad
            max_grad_lr = lr_s[-2]

    min_lr = min_lr * mul_factor
plt.xlabel('lr')
plt.ylabel('loss')
plt.plot(lr_s, loss_s)
plt.show()
print("grad:{} lr:{}".format(max_grad, max_grad_lr))


