#---------------step0:Common Definitaion-------------
import time

import torch
import random

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    #torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")

#------freeze backbone------
from collections.abc import Iterable
def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze


def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)


def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)


def set_freeze_by_idxs(model, idxs, freeze=True):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]
    num_child = len(list(model.children()))
    idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
    for idx, child in enumerate(model.children()):
        if idx not in idxs:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            print("idx:{} need_grad:{}".format(idx, param.requires_grad))


def freeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, True)

def unfreeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, False)

#torch.autograd.set_detect_anomaly(True)
# train hype parameter
batch_size = 4
lr = 1e-3
weight_decay = 5e-4
momentum = 0.9
pre_weight_file = "../PreTrain/darknet53_50.pth"

# train img parameter
img_sizes = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
base_img_size = 608 #基准尺度是608
anchor_boxes = [[49, 48], [88, 102], [128, 159], [176, 219], [224, 283], [269, 349], [309, 420], [348, 494], [419, 581]]
now_img_size = 416
#---------------step1:Dataset-------------------
import torch
#from COCO_DataSet import COCODataSet
from VOC_DataSet import VOCDataSet
dataSet = VOCDataSet(imgs_path="../DataSet/VOC2007/Train/JPEGImages",annotations_path="../DataSet/VOC2007/Train/Annotations",classes_file="../DataSet/VOC2007/class.data", class_num=20)
dataSet.setInputSize(now_img_size, anchor_boxes)
from torch.utils.data import DataLoader

#---------------step2:Model-------------------
from YOLO_V3_Model import YOLO_V3
YOLO = YOLO_V3(class_num=20).to(device=device)
YOLO.initialize_weights(pre_weight_file)
set_freeze_by_idxs(YOLO,[0, 1, 2, 3, 4])

#---------------step3:LossFunction-------------------
from YOLO_V3_LossFunction import YOLO_V3_Loss
loss_function = YOLO_V3_Loss(anchor_boxes=anchor_boxes).to(device=device)
loss_function.setImgSize(now_img_size, anchor_boxes)

#---------------step4:Optimizer-------------------
import torch.optim as optim
#optimizer_Adam = optim.Adam(YOLO.parameters(),lr=1e-4,weight_decay=0.005)
optimizer_SGD = optim.SGD(YOLO.parameters(),lr=lr,weight_decay=weight_decay, momentum=momentum)
#使用余弦退火动态调整学习率
#lr_reduce_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_Adam , T_max=20, eta_min=1e-4, last_epoch=-1)
#lr_reduce_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer_Adam, T_0=2, T_mult=2)

#--------------step5:Tensorboard Feature Map------------
import torch.nn as nn

def feature_map_visualize(img_data, writer):
    img_data = img_data.unsqueeze(0)
    img_grid = vutils.make_grid(img_data, normalize=True, scale_each=True)
    for i,m in enumerate(YOLO.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or \
                isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.AdaptiveAvgPool2d):
            img_data = m(img_data)
            x1 = img_data.transpose(0,1)
            img_grid = vutils.make_grid(x1, normalize=True, scale_each=True)
            writer.add_image('feature_map_' + str(i), img_grid)

#---------------step6:Train-------------------
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
epoch = 0
writer = SummaryWriter(logdir='./log', filename_suffix=' [' + str(epoch) + '~' + str(epoch + 100) + ']')

from tqdm import tqdm

while epoch <= 200 * dataSet.class_num:

    train_sum = int(dataSet.__len__() + 0.5)
    train_len = int(train_sum * 0.9)
    val_len = train_sum - train_len

    epoch_train_loss = 0
    epoch_val_loss = 0
    epoch_train_iou = 0
    epoch_val_iou = 0
    epoch_train_object_num = 0
    epoch_val_object_num = 0
    epoch_train_loss_coord = 0
    epoch_val_loss_coord = 0
    epoch_train_loss_confidence = 0
    epoch_val_loss_confidence = 0
    epoch_train_loss_classes = 0
    epoch_val_loss_classes = 0

    train_dataSet, val_dataSet = torch.utils.data.random_split(dataSet, [train_len, val_len])

    train_loader = DataLoader(train_dataSet, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_len = train_loader.__len__()
    YOLO.train()
    with tqdm(total=train_len) as tbar:
        for batch_index, batch_datas in enumerate(train_loader):
            optimizer_SGD.zero_grad()
            #for data_index in range(len(batch_datas)):
                #batch_datas[data_index] = batch_datas[data_index].float().to(device=device,non_blocking=True)
            #small_bounding_boxes, middle_bounding_boxes, big_bounding_boxes = YOLO(batch_datas[0].to(device=device,non_blocking=True))
            #loss = loss_function(small_bounding_boxes, middle_bounding_boxes, big_bounding_boxes, batch_datas[1].to(device=device,non_blocking=True), batch_datas[2].float().to(device=device,non_blocking=True), batch_datas[3].to(device=device,non_blocking=True), batch_datas[4].float().to(device=device,non_blocking=True), batch_datas[5].to(device=device,non_blocking=True), batch_datas[6].float().to(device=device,non_blocking=True))

            small_bounding_boxes, middle_bounding_boxes, big_bounding_boxes = YOLO(batch_datas[0].to(device=device))
            loss = loss_function(small_bounding_boxes, middle_bounding_boxes, big_bounding_boxes,
                                 batch_datas[1],
                                 batch_datas[2].float().to(device=device),
                                 batch_datas[3],
                                 batch_datas[4].float().to(device=device),
                                 batch_datas[5],
                                 batch_datas[6].float().to(device=device))
            print("loss 计算完毕")
            print(loss[0])
            batch_loss = loss[0] / batch_size
            '''
            batch_loss[0] = batch_loss[0] / batch_size
            batch_loss[1] = batch_loss[1] / batch_size
            batch_loss[2] = batch_loss[2] / batch_size
            '''
            epoch_train_loss_coord = epoch_train_loss_coord + loss[1]
            epoch_train_loss_confidence = epoch_train_loss_confidence + loss[2]
            epoch_train_loss_classes = epoch_train_loss_classes + loss[3]
            epoch_train_iou = epoch_train_iou + loss[4]
            epoch_train_object_num = epoch_train_object_num + loss[5]
            '''
            time_start = time.time()
            batch_loss[0].backward(retain_graph=True)
            time_end = time.time()
            print('loss_small:totally cost', time_end - time_start)
            time_start = time.time()
            batch_loss[1].backward(retain_graph=True)
            time_end = time.time()
            print('loss_middle:totally cost', time_end - time_start)
            time_start = time.time()
            batch_loss[2].backward()
            time_end = time.time()
            print('loss_big:totally cost', time_end - time_start)
            time_start = time.time()
            optimizer_SGD.step()
            time_end = time.time()
            print('sgd:totally cost', time_end - time_start)
            '''
            time_start = time.time()
            batch_loss.backward()
            optimizer_SGD.step()
            time_over = time.time()
            print("backward:{}".format(time_over - time_start))
            batch_loss = batch_loss.item()
            epoch_train_loss = epoch_train_loss + batch_loss
            #tbar.set_description("train: coord_loss:{} confidence_loss:{} class_loss:{} avg_iou:{}".format(round(loss[1], 4), round(loss[2], 4), round(loss[3], 4), round(loss[4] / loss[5], 4)), refresh=True)
            tbar.set_description(
                "train: coord_loss:{} confidence_loss:{} class_loss:{} avg_iou:{}".format(round(loss[1], 4),
                                                                                          round(loss[2], 4),
                                                                                          round(loss[3], 4),
                                                                                          0),
                refresh=True)
            tbar.update(1)

            #feature_map_visualize(train_data[0][0], writer)
            #print("batch_index : {} ; batch_loss : {}".format(batch_index, batch_loss))
        print("train-batch-mean loss:{} coord_loss:{} confidence_loss:{} class_loss:{} iou:{}".format(round(epoch_train_loss / train_len, 4), round(epoch_train_loss_coord / train_len, 4), round(epoch_train_loss_confidence / train_len, 4), round(epoch_train_loss_classes / train_len, 4), round(epoch_train_iou / epoch_train_object_num, 4)))

    #lr_reduce_scheduler.step()

    val_loader = DataLoader(val_dataSet, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_len = val_loader.__len__()
    YOLO.eval()
    with tqdm(total=val_len) as tbar:
        with torch.no_grad():
            for batch_index, batch_datas in enumerate(val_loader):
                for data_index in range(len(batch_datas)):
                    batch_datas[data_index] = batch_datas[data_index].float().to(device=device, non_blocking=True)
                small_bounding_boxes, middle_bounding_boxes, big_bounding_boxes = YOLO(batch_datas[0])
                loss = loss_function(small_bounding_boxes, middle_bounding_boxes, big_bounding_boxes,
                                     batch_datas[1],
                                     batch_datas[2],
                                     batch_datas[3],
                                     batch_datas[4],
                                     batch_datas[5],
                                     batch_datas[6])
                batch_loss = loss[0] / batch_size
                epoch_val_loss_coord = epoch_val_loss_coord + loss[1]
                epoch_val_loss_confidence = epoch_val_loss_confidence + loss[2]
                epoch_val_loss_classes = epoch_val_loss_classes + loss[3]
                epoch_val_iou = epoch_val_iou + loss[4]
                epoch_val_object_num = epoch_val_object_num + loss[5]
                batch_loss = batch_loss.item()
                epoch_val_loss = epoch_val_loss + batch_loss

                tbar.set_description("val: coord_loss:{} confidence_loss:{} class_loss:{} iou:{}".format(round(loss[1], 4), round(loss[2], 4), round(loss[3], 4), round(loss[4] / loss[5], 4)), refresh=True)
                tbar.update(1)

            # feature_map_visualize(train_data[0][0], writer)
            # print("batch_index : {} ; batch_loss : {}".format(batch_index, batch_loss))
        print("val-batch-mean loss:{} coord_loss:{} confidence_loss:{} class_loss:{} iou:{}".format(round(epoch_val_loss / val_len, 4), round(epoch_val_loss_coord / val_len, 4), round(epoch_val_loss_confidence / val_len, 4), round(epoch_val_loss_classes / val_len, 4), round(epoch_val_iou / epoch_val_object_num, 4)))

    epoch = epoch + 1

    if epoch % 2 == 0:
        dict = {}
        dict['model'] = YOLO.state_dict()
        dict['optim'] = optimizer_SGD
        dict['epoch'] = epoch

        transforms_seed = random.randint(0, 9)
        temp_input_size = img_sizes[transforms_seed]
        scale_factor = temp_input_size / base_img_size
        temp_anchors = []
        for anchor_box in anchor_boxes:
            temp_anchors.append([round(anchor_box[0] * scale_factor), round(anchor_box[1])])

        dataSet.setInputSize(temp_input_size, temp_anchors)
        loss_function.setImgSize(temp_input_size, temp_anchors)
        torch.save(dict, './YOLO_V3_' + str(epoch) + '.pth')
        writer.close()
        writer = SummaryWriter(logdir='log',filename_suffix='[' + str(epoch) + '~' + str(epoch + 100) + ']')

    print("epoch : {} ; loss : {}".format(epoch,{epoch_train_loss}))
    for name, layer in YOLO.named_parameters():
        writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

    writer.add_scalar('Train/Loss_sum', epoch_train_loss, epoch)
    writer.add_scalar('Train/Loss_coord', epoch_train_loss_coord, epoch)
    writer.add_scalar('Train/Loss_confidenct', epoch_train_loss_confidence, epoch)
    writer.add_scalar('Train/Loss_classes', epoch_train_loss_classes, epoch)
    writer.add_scalar('Train/Epoch_iou', epoch_train_iou / epoch_train_object_num, epoch)

    writer.add_scalar('Val/Loss_sum', epoch_val_loss, epoch)
    writer.add_scalar('Val/Loss_coord', epoch_val_loss_coord, epoch)
    writer.add_scalar('Val/Loss_confidenct', epoch_val_loss_confidence, epoch)
    writer.add_scalar('Val/Loss_classes', epoch_val_loss_classes, epoch)
    writer.add_scalar('Val/Epoch_iou', epoch_val_iou / epoch_val_object_num, epoch)

writer.close()
