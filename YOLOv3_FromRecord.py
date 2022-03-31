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

# train hype parameter
batch_size = 16
lr = 1e-4
weight_decay = 5e-4
momentum = 0.9
weight_file = "./Train/weights/YOLO_V3_200.pth"
param_dict = torch.load(weight_file, map_location=torch.device("cpu"))
min_val_loss = param_dict['min_val_loss']
lr = 1e-4
epoch = param_dict['epoch']
class_num = 20
batch_interval = 10
epoch_interval = 10
epoch_num = 200
epoch_unfreeze = 5
num_workers = 4
# train img parameter
img_sizes = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
base_img_size = 608 #基准尺度是608
anchor_boxes = [[21, 29], [35, 80], [61, 45],  [68, 143], [124, 95], [130, 229], [226, 339], [298, 174], [452, 384]]

#---------------step1:Dataset-------------------
import torch
from Train.VOC_DataSet import VOCDataSet
train_dataSet = VOCDataSet(imgs_path="../DataSet/VOC2007+2012/Train/JPEGImages",annotations_path="../DataSet/VOC2007+2012/Train/Annotations",classes_file="../DataSet/VOC2007+2012/class.data", is_train=True, class_num=class_num)
val_dataSet = VOCDataSet(imgs_path="../DataSet/VOC2007+2012/Val/JPEGImages",annotations_path="../DataSet/VOC2007+2012/Val/Annotations",classes_file="../DataSet/VOC2007+2012/class.data", is_train=False, class_num=class_num)
train_dataSet.setInputSize(base_img_size, anchor_boxes)
val_dataSet.setInputSize(base_img_size, anchor_boxes)

#---------------step2:Model-------------------
from Train.YOLOv3_Model import YOLO_V3
from utils.model import set_freeze_by_idxs, unfreeze_by_idxs
YOLO = YOLO_V3(class_num=class_num).to(device=device)
YOLO.load_state_dict(param_dict['model'])

#---------------step3:LossFunction-------------------
from Train.YOLOv3_LossFunction import YOLO_V3_Loss
loss_function = YOLO_V3_Loss(anchor_boxes=anchor_boxes, class_num=class_num).to(device=device)
loss_function.setImgSize(base_img_size, anchor_boxes)

#---------------step4:Optimizer-------------------
import torch.optim as optim
optimizer_SGD = optim.SGD(YOLO.parameters(),lr=lr,weight_decay=weight_decay, momentum=momentum)

#--------------step5:Tensorboard Feature Map------------
from utils.model import feature_map_visualize

#---------------step6:Train-------------------
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

if __name__ == '__main__':

    param_dict = {}
    writer = SummaryWriter(logdir='./Train/log', filename_suffix=' [' + str(epoch) + '~' + str(epoch + epoch_interval) + ']')

    while epoch <= epoch_num:

        epoch_train_loss = 0
        epoch_val_loss = 0

        epoch_train_small_iou_sum = 0
        epoch_train_middle_iou_sum = 0
        epoch_train_big_iou_sum = 0

        epoch_val_small_iou_sum = 0
        epoch_val_middle_iou_sum = 0
        epoch_val_big_iou_sum = 0

        epoch_train_small_positive_num = 0
        epoch_train_middle_positive_num = 0
        epoch_train_big_positive_num = 0

        epoch_val_small_positive_num = 0
        epoch_val_middle_positive_num = 0
        epoch_val_big_positive_num = 0

        epoch_train_small_loss_coord = 0
        epoch_train_middle_loss_coord = 0
        epoch_train_big_loss_coord = 0

        epoch_val_small_loss_coord = 0
        epoch_val_middle_loss_coord = 0
        epoch_val_big_loss_coord = 0

        epoch_train_small_positive_loss_confidence = 0
        epoch_train_middle_positive_loss_confidence = 0
        epoch_train_big_positive_loss_confidence = 0

        epoch_train_small_negative_loss_confidence = 0
        epoch_train_middle_negative_loss_confidence = 0
        epoch_train_big_negative_loss_confidence = 0

        epoch_val_small_positive_loss_confidence = 0
        epoch_val_middle_positive_loss_confidence = 0
        epoch_val_big_positive_loss_confidence = 0

        epoch_val_small_negative_loss_confidence = 0
        epoch_val_middle_negative_loss_confidence = 0
        epoch_val_big_negative_loss_confidence = 0

        epoch_train_small_loss_classes = 0
        epoch_train_middle_loss_classes = 0
        epoch_train_big_loss_classes = 0

        epoch_val_small_loss_classes = 0
        epoch_val_middle_loss_classes = 0
        epoch_val_big_loss_classes = 0

        train_loader = DataLoader(train_dataSet, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=True)
        train_len = train_loader.__len__()
        YOLO.train()
        with tqdm(total=train_len) as tbar:
            for batch_index, batch_datas in enumerate(train_loader):
                optimizer_SGD.zero_grad()
                for data_index in range(len(batch_datas)):
                    batch_datas[data_index] = batch_datas[data_index].to(device=device, non_blocking=True)
                # small_bounding_boxes, middle_bounding_boxes, big_bounding_boxes = YOLO(batch_datas[0].to(device=device,non_blocking=True))
                # loss = loss_function(small_bounding_boxes, middle_bounding_boxes, big_bounding_boxes, batch_datas[1].to(device=device,non_blocking=True), batch_datas[2].float().to(device=device,non_blocking=True), batch_datas[3].to(device=device,non_blocking=True), batch_datas[4].float().to(device=device,non_blocking=True), batch_datas[5].to(device=device,non_blocking=True), batch_datas[6].float().to(device=device,non_blocking=True))

                small_bounding_boxes, middle_bounding_boxes, big_bounding_boxes = YOLO(batch_datas[0])

                batch_loss, batch_small_loss_coord, batch_small_positive_loss_confidence, batch_small_loss_classes, batch_small_negative_loss_confidence, \
                batch_middle_loss_coord, batch_middle_positive_loss_confidence, batch_middle_loss_classes, batch_middle_negative_loss_confidence, \
                batch_big_loss_coord, batch_big_positive_loss_confidence, batch_big_loss_classes, batch_big_negative_loss_confidence, \
                batch_small_iou_sum, batch_middle_iou_sum, batch_big_iou_sum, batch_small_positive_num, batch_middle_positive_num, batch_big_positive_num = loss_function(
                    small_bounding_boxes, middle_bounding_boxes, big_bounding_boxes,
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

                epoch_train_small_loss_coord = epoch_train_small_loss_coord + batch_small_loss_coord
                epoch_train_middle_loss_coord = epoch_train_middle_loss_coord + batch_middle_loss_coord
                epoch_train_big_loss_coord = epoch_train_big_loss_coord + batch_big_loss_coord

                epoch_train_small_positive_loss_confidence = epoch_train_small_positive_loss_confidence + batch_small_positive_loss_confidence
                epoch_train_middle_positive_loss_confidence = epoch_train_middle_positive_loss_confidence + batch_middle_positive_loss_confidence
                epoch_train_big_positive_loss_confidence = epoch_train_big_positive_loss_confidence + batch_big_positive_loss_confidence

                epoch_train_small_negative_loss_confidence = epoch_train_small_negative_loss_confidence + batch_small_negative_loss_confidence
                epoch_train_middle_negative_loss_confidence = epoch_train_middle_negative_loss_confidence + batch_middle_negative_loss_confidence
                epoch_train_big_negative_loss_confidence = epoch_train_big_negative_loss_confidence + batch_big_negative_loss_confidence

                epoch_train_small_loss_classes = epoch_train_small_loss_classes + batch_small_loss_classes
                epoch_train_middle_loss_classes = epoch_train_middle_loss_classes + batch_middle_loss_classes
                epoch_train_big_loss_classes = epoch_train_big_loss_classes + batch_big_loss_classes

                epoch_train_small_iou_sum = epoch_train_small_iou_sum + batch_small_iou_sum
                epoch_train_middle_iou_sum = epoch_train_middle_iou_sum + batch_middle_iou_sum
                epoch_train_big_iou_sum = epoch_train_big_iou_sum + batch_big_iou_sum

                epoch_train_small_positive_num = epoch_train_small_positive_num + batch_small_positive_num
                epoch_train_middle_positive_num = epoch_train_middle_positive_num + batch_middle_positive_num
                epoch_train_big_positive_num = epoch_train_big_positive_num + batch_big_positive_num

                small_mean_iou = batch_small_iou_sum / batch_small_positive_num if batch_small_positive_num != 0 else 0
                middle_mean_iou = batch_middle_iou_sum / batch_middle_positive_num if batch_middle_positive_num != 0 else 0
                big_mean_iou = batch_big_iou_sum / batch_big_positive_num if batch_big_positive_num != 0 else 0

                batch_loss.backward()
                optimizer_SGD.step()

                batch_loss = batch_loss.item()
                epoch_train_loss = epoch_train_loss + batch_loss
                tbar.set_description(
                    "train: small_coord_loss:{} small_confidence_loss positive:{} negative:{} small_class_loss:{} small_avg_iou:{} middle_coord_loss:{} middle_confidence_loss positive:{} negative:{} middle_class_loss:{} middle_avg_iou:{} big_coord_loss:{} big_confidence_loss positive:{} negative:{} big_class_loss:{} big_avg_iou:{}".format(
                        round(batch_small_loss_coord, 4),
                        round(batch_small_positive_loss_confidence, 4),
                        round(batch_small_negative_loss_confidence, 4),
                        round(batch_small_loss_classes, 4),
                        round(small_mean_iou, 4),
                        round(batch_middle_loss_coord, 4),
                        round(batch_middle_positive_loss_confidence, 4),
                        round(batch_middle_negative_loss_confidence, 4),
                        round(batch_middle_loss_classes, 4),
                        round(middle_mean_iou, 4),
                        round(batch_big_loss_coord, 4),
                        round(batch_big_positive_loss_confidence, 4),
                        round(batch_big_negative_loss_confidence, 4),
                        round(batch_big_loss_classes, 4),
                        round(big_mean_iou, 4)),
                    refresh=True)
                tbar.update(1)

                if batch_index != 0 and batch_index % batch_interval == 0:

                    transforms_seed = random.randint(0, 9)
                    temp_input_size = img_sizes[transforms_seed]
                    scale_factor = temp_input_size / base_img_size
                    temp_anchors = []
                    for anchor_box in anchor_boxes:
                        temp_anchors.append([round(anchor_box[0] * scale_factor), round(anchor_box[1])])

                    train_dataSet.setInputSize(temp_input_size, temp_anchors)
                    loss_function.setImgSize(temp_input_size, temp_anchors)

                # feature_map_visualize(train_data[0][0], writer)
            print(
                "train-batch-mean loss:{} small_coord_loss:{} small_confidence_loss positive:{} negative:{} small_class_loss:{} small_iou:{} middle_coord_loss:{} middle_confidence_loss positive:{} negative:{} middle_class_loss:{} middle_iou:{} big_coord_loss:{} big_confidence_loss positive:{} negative:{} big_class_loss:{} big_iou{}".format(
                    round(epoch_train_loss / train_len, 4),
                    round(epoch_train_small_loss_coord / train_len, 4),
                    round(epoch_train_small_positive_loss_confidence / train_len, 4),
                    round(epoch_train_small_negative_loss_confidence / train_len, 4),
                    round(epoch_train_small_loss_classes / train_len, 4),
                    round(epoch_train_small_iou_sum / epoch_train_small_positive_num, 4),
                    round(epoch_train_middle_loss_coord / train_len, 4),
                    round(epoch_train_middle_positive_loss_confidence / train_len, 4),
                    round(epoch_train_middle_negative_loss_confidence / train_len, 4),
                    round(epoch_train_middle_loss_classes / train_len, 4),
                    round(epoch_train_middle_iou_sum / epoch_train_middle_positive_num, 4),
                    round(epoch_train_big_loss_coord / train_len, 4),
                    round(epoch_train_big_positive_loss_confidence / train_len, 4),
                    round(epoch_train_big_negative_loss_confidence / train_len, 4),
                    round(epoch_train_big_loss_classes / train_len, 4),
                    round(epoch_train_big_iou_sum / epoch_train_big_positive_num, 4)))

        # lr_reduce_scheduler.step()

        val_loader = DataLoader(val_dataSet, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_len = val_loader.__len__()
        YOLO.eval()
        with tqdm(total=val_len) as tbar:
            with torch.no_grad():
                for batch_index, batch_datas in enumerate(val_loader):
                    for data_index in range(len(batch_datas)):
                        batch_datas[data_index] = batch_datas[data_index].to(device=device, non_blocking=True)
                    small_bounding_boxes, middle_bounding_boxes, big_bounding_boxes = YOLO(batch_datas[0])
                    batch_loss, batch_small_loss_coord, batch_small_positive_loss_confidence, batch_small_loss_classes, batch_small_negative_loss_confidence, \
                    batch_middle_loss_coord, batch_middle_positive_loss_confidence, batch_middle_loss_classes, batch_middle_negative_loss_confidence, \
                    batch_big_loss_coord, batch_big_positive_loss_confidence, batch_big_loss_classes, batch_big_negative_loss_confidence, \
                    batch_small_iou_sum, batch_middle_iou_sum, batch_big_iou_sum, batch_small_positive_num, batch_middle_positive_num, batch_big_positive_num = loss_function(
                        small_bounding_boxes, middle_bounding_boxes, big_bounding_boxes,
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

                    epoch_val_small_loss_coord = epoch_val_small_loss_coord + batch_small_loss_coord
                    epoch_val_middle_loss_coord = epoch_val_middle_loss_coord + batch_middle_loss_coord
                    epoch_val_big_loss_coord = epoch_val_big_loss_coord + batch_big_loss_coord

                    epoch_val_small_positive_loss_confidence = epoch_val_small_positive_loss_confidence + batch_small_positive_loss_confidence
                    epoch_val_middle_positive_loss_confidence = epoch_val_middle_positive_loss_confidence + batch_middle_positive_loss_confidence
                    epoch_val_big_positive_loss_confidence = epoch_val_big_positive_loss_confidence + batch_big_positive_loss_confidence

                    epoch_val_small_negative_loss_confidence = epoch_val_small_negative_loss_confidence + batch_small_negative_loss_confidence
                    epoch_val_middle_negative_loss_confidence = epoch_val_middle_negative_loss_confidence + batch_middle_negative_loss_confidence
                    epoch_val_big_negative_loss_confidence = epoch_val_big_negative_loss_confidence + batch_big_negative_loss_confidence

                    epoch_val_small_loss_classes = epoch_val_small_loss_classes + batch_small_loss_classes
                    epoch_val_middle_loss_classes = epoch_val_middle_loss_classes + batch_middle_loss_classes
                    epoch_val_big_loss_classes = epoch_val_big_loss_classes + batch_big_loss_classes

                    epoch_val_small_iou_sum = epoch_val_small_iou_sum + batch_small_iou_sum
                    epoch_val_middle_iou_sum = epoch_val_middle_iou_sum + batch_middle_iou_sum
                    epoch_val_big_iou_sum = epoch_val_big_iou_sum + batch_big_iou_sum

                    epoch_val_small_positive_num = epoch_val_small_positive_num + batch_small_positive_num
                    epoch_val_middle_positive_num = epoch_val_middle_positive_num + batch_middle_positive_num
                    epoch_val_big_positive_num = epoch_val_big_positive_num + batch_big_positive_num

                    small_mean_iou = batch_small_iou_sum / batch_small_positive_num if batch_small_positive_num != 0 else 0
                    middle_mean_iou = batch_middle_iou_sum / batch_middle_positive_num if batch_middle_positive_num != 0 else 0
                    big_mean_iou = batch_big_iou_sum / batch_big_positive_num if batch_big_positive_num != 0 else 0

                    batch_loss = batch_loss.item()
                    epoch_val_loss = epoch_val_loss + batch_loss
                    tbar.set_description(
                        "val: small_coord_loss:{} small_confidence_loss positive:{} negative:{} small_class_loss:{} small_avg_iou:{} middle_coord_loss:{} middle_confidence_loss positive:{} negative:{} middle_class_loss:{} middle_avg_iou:{} big_coord_loss:{} big_confidence_loss positive:{} negative:{} big_class_loss:{} big_avg_iou:{}".format(
                            round(batch_small_loss_coord, 4),
                            round(batch_small_positive_loss_confidence, 4),
                            round(batch_small_negative_loss_confidence, 4),
                            round(batch_small_loss_classes, 4),
                            round(small_mean_iou, 4),
                            round(batch_middle_loss_coord, 4),
                            round(batch_middle_positive_loss_confidence, 4),
                            round(batch_middle_negative_loss_confidence, 4),
                            round(batch_middle_loss_classes, 4),
                            round(middle_mean_iou, 4),
                            round(batch_big_loss_coord, 4),
                            round(batch_big_positive_loss_confidence, 4),
                            round(batch_big_negative_loss_confidence, 4),
                            round(batch_big_loss_classes, 4),
                            round(big_mean_iou, 4)),
                        refresh=True)
                    tbar.update(1)

                    # feature_map_visualize(train_data[0][0], writer)
                print(
                    "val-batch-mean loss:{} small_coord_loss:{} small_confidence_loss positive:{} negative:{} small_class_loss:{} small_iou:{} middle_coord_loss:{} middle_confidence_loss positive:{} negative:{} middle_class_loss:{} middle_iou:{} big_coord_loss:{} big_confidence_loss positive:{} negative:{} big_class_loss:{} big_iou{}".format(
                        round(epoch_val_loss / val_len, 4),
                        round(epoch_val_small_loss_coord / val_len, 4),
                        round(epoch_val_small_positive_loss_confidence / val_len, 4),
                        round(epoch_val_small_negative_loss_confidence / val_len, 4),
                        round(epoch_val_small_loss_classes / val_len, 4),
                        round(epoch_val_small_iou_sum / epoch_val_small_positive_num, 4),
                        round(epoch_val_middle_loss_coord / val_len, 4),
                        round(epoch_val_middle_positive_loss_confidence / val_len, 4),
                        round(epoch_val_middle_negative_loss_confidence / val_len, 4),
                        round(epoch_val_middle_loss_classes / val_len, 4),
                        round(epoch_val_middle_iou_sum / epoch_val_middle_positive_num, 4),
                        round(epoch_val_big_loss_coord / val_len, 4),
                        round(epoch_val_big_positive_loss_confidence / val_len, 4),
                        round(epoch_val_big_negative_loss_confidence / val_len, 4),
                        round(epoch_val_big_loss_classes / val_len, 4),
                        round(epoch_val_big_iou_sum / epoch_val_big_positive_num, 4)))
        epoch = epoch + 1

        if epoch == epoch_unfreeze:
            unfreeze_by_idxs(YOLO, [0, 1, 2, 3, 4])

        if min_val_loss > epoch_val_loss:
            min_val_loss = epoch_val_loss
            param_dict['min_val_loss'] = min_val_loss
            param_dict['min_loss_model'] = YOLO.state_dict()

        print("epoch : {} ; loss : {}".format(epoch, {epoch_val_loss}))

        # ------------怎么保存？？？？？？？------------
        if epoch % epoch_interval == 0:

            param_dict['model'] = YOLO.state_dict()
            param_dict['optim'] = optimizer_SGD
            param_dict['epoch'] = epoch
            torch.save(param_dict, './Train/weights/YOLO_V3_' + str(epoch) + '.pth')
            writer.close()
            writer = SummaryWriter(logdir='log',
                                   filename_suffix='[' + str(epoch) + '~' + str(epoch + epoch_interval) + ']')

        # for name, layer in YOLO.named_parameters():
        # writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        # writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

        writer.add_scalar('Train/Loss_sum', epoch_train_loss, epoch)

        writer.add_scalar('Train/Loss_small_coord', epoch_train_small_loss_coord, epoch)
        writer.add_scalar('Train/Loss_middle_coord', epoch_train_middle_loss_coord, epoch)
        writer.add_scalar('Train/Loss_big_coord', epoch_train_big_loss_coord, epoch)

        writer.add_scalar('Train/Loss_small_positive_confidenct', epoch_train_small_positive_loss_confidence, epoch)
        writer.add_scalar('Train/Loss_middle_positive_confidenct', epoch_train_middle_positive_loss_confidence, epoch)
        writer.add_scalar('Train/Loss_big_positive_confidenct', epoch_train_big_positive_loss_confidence, epoch)

        writer.add_scalar('Train/Loss_small_negative_confidenct', epoch_train_small_negative_loss_confidence, epoch)
        writer.add_scalar('Train/Loss_middle_negative_confidenct', epoch_train_middle_negative_loss_confidence, epoch)
        writer.add_scalar('Train/Loss_big_negative_confidenct', epoch_train_big_negative_loss_confidence, epoch)

        writer.add_scalar('Train/Loss_small_classes', epoch_train_small_loss_classes, epoch)
        writer.add_scalar('Train/Loss_middle_classes', epoch_train_middle_loss_classes, epoch)
        writer.add_scalar('Train/Loss_big_classes', epoch_train_big_loss_classes, epoch)

        writer.add_scalar('Train/Epoch_small_iou', epoch_train_small_iou_sum / epoch_train_small_positive_num, epoch)
        writer.add_scalar('Train/Epoch_middle_iou', epoch_train_middle_iou_sum / epoch_train_middle_positive_num, epoch)
        writer.add_scalar('Train/Epoch_big_iou', epoch_train_big_iou_sum / epoch_train_big_positive_num, epoch)

        writer.add_scalar('Val/Loss_sum', epoch_val_loss, epoch)

        writer.add_scalar('Val/Loss_small_coord', epoch_val_small_loss_coord, epoch)
        writer.add_scalar('Val/Loss_middle_coord', epoch_val_middle_loss_coord, epoch)
        writer.add_scalar('Val/Loss_big_coord', epoch_val_big_loss_coord, epoch)

        writer.add_scalar('Val/Loss_small_positive_confidenct', epoch_val_small_positive_loss_confidence, epoch)
        writer.add_scalar('Val/Loss_middle_positive_confidenct', epoch_val_middle_positive_loss_confidence, epoch)
        writer.add_scalar('Val/Loss_big_positive_confidenct', epoch_val_big_positive_loss_confidence, epoch)

        writer.add_scalar('Val/Loss_small_negative_confidenct', epoch_val_small_negative_loss_confidence, epoch)
        writer.add_scalar('Val/Loss_middle_negative_confidenct', epoch_val_middle_negative_loss_confidence, epoch)
        writer.add_scalar('Val/Loss_big_negative_confidenct', epoch_val_big_negative_loss_confidence, epoch)

        writer.add_scalar('Val/Loss_small_classes', epoch_val_small_loss_classes, epoch)
        writer.add_scalar('Val/Loss_middle_classes', epoch_val_middle_loss_classes, epoch)
        writer.add_scalar('Val/Loss_big_classes', epoch_val_big_loss_classes, epoch)

        writer.add_scalar('Val/Epoch_small_iou', epoch_val_small_iou_sum / epoch_val_small_positive_num, epoch)
        writer.add_scalar('Train/Epoch_middle_iou', epoch_val_middle_iou_sum / epoch_val_middle_positive_num, epoch)
        writer.add_scalar('Val/Epoch_big_iou', epoch_val_big_iou_sum / epoch_val_big_positive_num, epoch)

        writer.add_scalar('Val/Epoch_small_iou', epoch_val_small_iou_sum / epoch_val_small_positive_num, epoch)
        writer.add_scalar('Val/Epoch_middle_iou', epoch_val_middle_iou_sum / epoch_val_middle_positive_num, epoch)
        writer.add_scalar('Val/Epoch_big_iou', epoch_val_big_iou_sum / epoch_val_big_positive_num, epoch)

    writer.close()