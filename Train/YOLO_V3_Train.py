#---------------step0:Common Definitaion-------------
import torch
import random
from datetime import datetime
random.seed(datetime.now())

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    #torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")

#torch.autograd.set_detect_anomaly(True)
# train hype parameter
batch_size = 16
lr = 1e-3
weight_decay = 5e-4
momentum = 0.9
pre_weight_file = "../PreTrain/darknet53_901.pth"
class_num = 20
epoch_interval = 50
epoch_num = 200
num_workers = 4
min_val_loss = 9999999999
# train img parameter
img_sizes = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
base_img_size = 608 #基准尺度是608

anchor_boxes = [[10, 11], [15, 28], [36, 22], [30, 60], [61, 125], [67, 46], [129, 88], [162, 211], [391, 336]]
#anchor_boxes = [[7, 10], [14, 30], [23, 14], [30, 61], [46, 30], [61, 118], [98, 61], [148, 188], [350, 330]]

#---------------step1:Dataset-------------------
import torch

from VOC_DataSet import VOCDataSet
train_dataSet = VOCDataSet(imgs_path="../DataSet/VOC2007+2012/Train/JPEGImages",annotations_path="../DataSet/VOC2007+2012/Train/Annotations",classes_file="../DataSet/VOC2007+2012/class.data", is_train=True, class_num=class_num)
val_dataSet = VOCDataSet(imgs_path="../DataSet/VOC2007+2012/Val/JPEGImages",annotations_path="../DataSet/VOC2007+2012/Val/Annotations",classes_file="../DataSet/VOC2007+2012/class.data", is_train=False, class_num=class_num)
train_dataSet.setInputSize(base_img_size, anchor_boxes)
val_dataSet.setInputSize(base_img_size, anchor_boxes)
#from COCO_DataSet import COCODataSet
#dataSet = COCODataSet(imgs_path="../DataSet/COCO2017/Train/JPEGImages",txts_path="../DataSet/COCO2017/Train/Labels", class_num=80)

#---------------step2:Model-------------------
from YOLO_V3_Model import YOLO_V3
from model import set_freeze_by_idxs
YOLO = YOLO_V3(class_num=80).to(device=device)
YOLO.initialize_weights(pre_weight_file)
set_freeze_by_idxs(YOLO,[0, 1, 2, 3, 4])

#---------------step3:LossFunction-------------------
from YOLO_V3_LossFunction import YOLO_V3_Loss
loss_function = YOLO_V3_Loss(anchor_boxes=anchor_boxes, class_num=class_num).to(device=device)
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
import torchvision.utils as vutils
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
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

if __name__ == '__main__':

    epoch = 0
    param_dict = {}
    writer = SummaryWriter(logdir='./log', filename_suffix=' [' + str(epoch) + '~' + str(epoch + epoch_interval) + ']')

    while epoch <= epoch_num:

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

        train_loader = DataLoader(train_dataSet, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        train_len = train_loader.__len__()
        YOLO.train()
        with tqdm(total=train_len) as tbar:
            for batch_index, batch_datas in enumerate(train_loader):
                optimizer_SGD.zero_grad()
                for data_index in range(len(batch_datas)):
                    batch_datas[data_index] = batch_datas[data_index].to(device=device,non_blocking=True)
                #small_bounding_boxes, middle_bounding_boxes, big_bounding_boxes = YOLO(batch_datas[0].to(device=device,non_blocking=True))
                #loss = loss_function(small_bounding_boxes, middle_bounding_boxes, big_bounding_boxes, batch_datas[1].to(device=device,non_blocking=True), batch_datas[2].float().to(device=device,non_blocking=True), batch_datas[3].to(device=device,non_blocking=True), batch_datas[4].float().to(device=device,non_blocking=True), batch_datas[5].to(device=device,non_blocking=True), batch_datas[6].float().to(device=device,non_blocking=True))

                small_bounding_boxes, middle_bounding_boxes, big_bounding_boxes = YOLO(batch_datas[0])
                loss = loss_function(small_bounding_boxes, middle_bounding_boxes, big_bounding_boxes,
                                     batch_datas[1],
                                     batch_datas[2],
                                     batch_datas[3],
                                     batch_datas[4],
                                     batch_datas[5],
                                     batch_datas[6],
                                     batch_datas[7],
                                     batch_datas[8],
                                     batch_datas[9],
                                     batch_datas[10],
                                     batch_datas[11],
                                     batch_datas[12],
                                     batch_datas[13],
                                     batch_datas[14],
                                     batch_datas[15],
                                     )

                batch_loss = loss[0]

                epoch_train_loss_coord = epoch_train_loss_coord + loss[1]
                epoch_train_loss_confidence = epoch_train_loss_confidence + loss[2]
                epoch_train_loss_classes = epoch_train_loss_classes + loss[3]
                epoch_train_iou = epoch_train_iou + loss[4]
                epoch_train_object_num = epoch_train_object_num + loss[5]

                batch_loss.backward()
                optimizer_SGD.step()

                batch_loss = batch_loss.item()
                epoch_train_loss = epoch_train_loss + batch_loss
                tbar.set_description(
                    "train: coord_loss:{} confidence_loss:{} class_loss:{} avg_iou:{}".format(round(loss[1], 4),
                                                                                              round(loss[2], 4),
                                                                                              round(loss[3], 4),
                                                                                              round(loss[4] / loss[5], 4)),
                    refresh=True)
                tbar.update(1)

                #feature_map_visualize(train_data[0][0], writer)
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

        if min_val_loss > epoch_val_loss:
            min_val_loss = epoch_val_loss
            param_dict['min_val_loss'] = min_val_loss
            param_dict['min_loss_model'] = YOLO.state_dict()

        if epoch % epoch_interval == 0:
            param_dict['model'] = YOLO.state_dict()
            param_dict['optim'] = optimizer_SGD
            param_dict['epoch'] = epoch
            torch.save(param_dict, './weights/YOLO_V1_PreTrain_' + str(epoch) + '.pth')
            writer.close()
            writer = SummaryWriter(logdir='log', filename_suffix='[' + str(epoch) + '~' + str(epoch + epoch_interval) + ']')
        print("epoch : {} ; loss : {}".format(epoch, {epoch_train_loss}))

        # ------------怎么保存？？？？？？？------------
        if epoch % 10 == 0:
            transforms_seed = random.randint(0, 9)
            temp_input_size = img_sizes[transforms_seed]
            scale_factor = temp_input_size / base_img_size
            temp_anchors = []
            for anchor_box in anchor_boxes:
                temp_anchors.append([round(anchor_box[0] * scale_factor), round(anchor_box[1])])

            train_dataSet.setInputSize(temp_input_size, temp_anchors)
            val_dataSet.setInputSize(temp_input_size, temp_anchors)
            loss_function.setImgSize(temp_input_size, temp_anchors)

        if min_val_loss > epoch_val_loss:
            min_val_loss = epoch_val_loss
            param_dict['min_val_loss'] = min_val_loss
            param_dict['min_loss_model'] = YOLO.state_dict()

        if epoch % epoch_interval == 0:
            dict = {}
            dict['model'] = YOLO.state_dict()
            dict['optim'] = optimizer_SGD
            dict['epoch'] = epoch

            torch.save(dict, './YOLO_V3_' + str(epoch) + '.pth')
            writer.close()
            writer = SummaryWriter(logdir='log',filename_suffix='[' + str(epoch) + '~' + str(epoch + epoch_interval) + ']')

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
