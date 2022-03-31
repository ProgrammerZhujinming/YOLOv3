#---------------step0:Common Definition-----------------
import torch
import torch.nn as nn
from utils.model import accuracy
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

lr = 3e-4
img_size = 256
momentum = 0.9
batch_size = 64
epoch_num = 500
weight_decay = 5e-4
min_val_loss = 9999999999
epoch_interval = 10
class_num = 80
num_workers = 2

#---------------step1:Dataset-------------------
from PreTrain.COCO_Classify import  coco_classify
train_dataSet = coco_classify(imgs_path="./DataSet/COCO2017/Train/Images", txts_path= "./DataSet/COCO2017/Train/Labels", is_train=True)
val_dataSet = coco_classify(imgs_path="./DataSet/COCO2017/Val/Images", txts_path= "./DataSet/COCO2017/Val/Labels", is_train=False)

#---------------step2:Model-------------------
from PreTrain.DarkNet53 import DarkNet53
darkNet53 = DarkNet53(class_num=class_num).to(device=device)
darkNet53.weight_init()

#---------------step3:LossFunction-------------------
loss_function = nn.CrossEntropyLoss().to(device=device)

#---------------step4:Optimizer-------------------
import torch.optim as optim
optimizer = optim.SGD(darkNet53.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

#--------------step5:Tensorboard Feature Map------------
from utils.model import feature_map_visualize

#---------------step6:Train-------------------
from tqdm import tqdm
from tensorboardX import SummaryWriter
if __name__ == "__main__":

    epoch = 0

    param_dict = {}

    writer = SummaryWriter(logdir='./PreTrain/log', filename_suffix=' [' + str(epoch) + '~' + str(epoch + epoch_interval) + ']')

    while epoch < epoch_num:

        epoch_train_loss = 0
        epoch_val_loss = 0
        epoch_train_top1_acc = 0
        epoch_train_top5_acc = 0
        epoch_val_top1_acc = 0
        epoch_val_top5_acc = 0

        train_loader = DataLoader(dataset=train_dataSet, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=True)
        train_len = train_loader.__len__()
        darkNet53.train()
        with tqdm(total=train_len) as tbar:

            for batch_index, batch_train in enumerate(train_loader):
                train_data = batch_train[0].float().to(device=device, non_blocking=True)
                label_data = batch_train[1].long().to(device=device, non_blocking=True)
                net_out = darkNet53(train_data)
                loss = loss_function(net_out, label_data)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_loss = loss.item() * batch_size
                epoch_train_loss = epoch_train_loss + batch_loss

                # 计算准确率
                net_out = net_out.detach()
                [top1_acc, top5_acc] = accuracy(net_out, label_data)
                top1_acc = top1_acc.item()
                top5_acc = top5_acc.item()

                epoch_train_top1_acc = epoch_train_top1_acc + top1_acc
                epoch_train_top5_acc = epoch_train_top5_acc + top5_acc

                tbar.set_description(
                    "train: class_loss:{} top1-acc:{} top5-acc:{}".format(loss.item(), round(top1_acc, 4),
                                                                          round(top5_acc, 4), refresh=True))
                tbar.update(1)

                # feature_map_visualize(darkNet53, train_data[0][0], writer)
                # print("batch_index : {} ; batch_loss : {}".format(batch_index, batch_loss))
            print(
                "train-mean: batch_loss:{} batch_top1_acc:{} batch_top5_acc:{}".format(round(epoch_train_loss / train_loader.__len__(), 4), round(
                    epoch_train_top1_acc / train_loader.__len__(), 4), round(
                    epoch_train_top5_acc / train_loader.__len__(), 4)))

        # lr_reduce_scheduler.step()

        val_loader = DataLoader(dataset=val_dataSet, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                pin_memory=True)
        val_len = val_loader.__len__()
        darkNet53.eval()
        with tqdm(total=val_len) as tbar:
            with torch.no_grad():
                for batch_index, batch_train in enumerate(val_loader):
                    train_data = batch_train[0].float().to(device=device, non_blocking=True)
                    label_data = batch_train[1].long().to(device=device, non_blocking=True)
                    net_out = darkNet53(train_data)
                    loss = loss_function(net_out, label_data)
                    batch_loss = loss.item() * batch_size
                    epoch_val_loss = epoch_val_loss + batch_loss

                    # 计算准确率
                    net_out = net_out.detach()
                    [top1_acc, top5_acc] = accuracy(net_out, label_data)
                    top1_acc = top1_acc.item()
                    top5_acc = top5_acc.item()

                    epoch_val_top1_acc = epoch_val_top1_acc + top1_acc
                    epoch_val_top5_acc = epoch_val_top5_acc + top5_acc

                    tbar.set_description(
                        "val: class_loss:{} top1-acc:{} top5-acc:{}".format(loss.item(), round(top1_acc, 4),
                                                                            round(top5_acc, 4), refresh=True))
                    tbar.update(1)

                # feature_map_visualize(darkNet53, train_data[0][0], writer)
                # print("batch_index : {} ; batch_loss : {}".format(batch_index, batch_loss))
            print(
                "train-mean: batch_loss:{} batch_top1_acc:{} batch_top5_acc:{}".format(round(epoch_val_loss / val_loader.__len__(), 4), round(
                    epoch_val_top1_acc / val_loader.__len__(), 4), round(
                    epoch_val_top5_acc / val_loader.__len__(), 4)))
        epoch = epoch + 1

        if min_val_loss > epoch_val_loss:
            min_val_loss = epoch_val_loss
            param_dict['min_val_loss'] = min_val_loss
            param_dict['min_loss_model'] = darkNet53.state_dict()

        if epoch % epoch_interval == 0:
            param_dict['model'] = darkNet53.state_dict()
            param_dict['optim'] = optimizer
            param_dict['epoch'] = epoch
            torch.save(param_dict, './PreTrain/weights/Darknet-53_' + str(epoch) + '.pth')
            writer.close()
            writer = SummaryWriter(logdir='./PreTrain/log', filename_suffix='[' + str(epoch) + '~' + str(epoch + epoch_interval) + ']')
        print("epoch : {} ; train-loss : {}".format(epoch, {epoch_train_loss}))

        for i, (name, layer) in enumerate(darkNet53.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name + '_grad', layer, epoch)

        writer.add_scalar('Train/Loss_sum', epoch_train_loss, epoch)
        writer.add_scalar('Train/Top_Acc_1', epoch_train_top1_acc / train_loader.__len__(), epoch)
        writer.add_scalar('Train/Top_Acc_5', epoch_train_top5_acc / train_loader.__len__(), epoch)
        writer.add_scalar('Val/Loss_sum', epoch_val_loss, epoch)
        writer.add_scalar('Val/Top_Acc_1', epoch_val_top1_acc / val_loader.__len__(), epoch)
        writer.add_scalar('Val/Top_Acc_5', epoch_val_top5_acc / val_loader.__len__(), epoch)
    writer.close()