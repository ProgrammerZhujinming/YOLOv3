import time
import torch.nn as nn
import math
import torch

class YOLO_V3_Loss(nn.Module):

    def __init__(self, anchor_boxes, small_downsample=8, middle_downsample=16, big_downsample=32, class_num=80, B=3, l_coord=50, l_noobj=0.5):
        # 有物体的box损失权重设为l_coord,没有物体的box损失权重设置为l_noobj
        super(YOLO_V3_Loss, self).__init__()
        self.B = B
        self.class_num = class_num
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.anchor_boxes = anchor_boxes
        self.small_downsmaple = small_downsample
        self.middle_downsmaple = middle_downsample
        self.big_downsmaple = big_downsample

    def iou(self, predict_coord, ground_coord):  # 计算两个box的IoU值 存储格式 xmin ymin xmax ymax

        predict_Area = (predict_coord[2] - predict_coord[0]) * (predict_coord[3] - predict_coord[1])
        ground_Area = (ground_coord[2] - ground_coord[0]) * (ground_coord[3] - ground_coord[1])

        CrossLX = max(predict_coord[0], ground_coord[0])
        CrossRX = min(predict_coord[2], ground_coord[2])
        CrossUY = max(predict_coord[1], ground_coord[1])
        CrossDY = min(predict_coord[3], ground_coord[3])

        if CrossRX < CrossLX or CrossDY < CrossUY:  # 没有交集
            return 0

        interSection = (CrossRX - CrossLX) * (CrossDY - CrossUY)

        return interSection / (predict_Area + ground_Area - interSection)



    def forward(self, samll_bounding_boxes, middle_bounding_boxes, big_bounding_boxes, small_ground_truth, small_positive_modulus, small_anchor_mark_positive, small_anchor_mark_negative, small_positive_modulus_mark, middle_ground_truth, middle_positive_modulus, middle_anchor_mark_positive, middle_anchor_mark_negative, middle_positive_modulus_mark, big_ground_truth, big_positive_modulus, big_anchor_mark_positive, big_anchor_mark_negative, big_positive_modulus_mark):  # 输入是 S * S * ( 2 * B + Classes)
        # 定义三个计算损失的变量 正样本定位损失 样本置信度损失 样本类别损失
        batch_size = len(samll_bounding_boxes[0])
        loss = 0
        loss_coord = 0
        loss_confidence = 0
        loss_classes = 0
        iou_sum = 0
        object_num = 0
        #mse_loss = nn.MSELoss()
        #bce_loss = nn.BCELoss()
        positives_num = 0
        negatives_num = 0
        bce_loss = nn.BCEWithLogitsLoss()

        small_grid_feature_size = round(self.img_size / self.small_downsmaple)
        middle_grid_feature_size = round(self.img_size / self.middle_downsmaple)
        big_grid_feature_size = round(self.img_size / self.big_downsmaple)

        time_start = time.time()

        # ground_size, batch_size, width, height, 3个anchor
        # small_ground_truth = small_ground_truth.permute(4, 0, 1, 2, 3)
        # samll_bounding_boxes = samll_bounding_boxes.permute(4, 0, 1, 2, 3)
        #<=================small loss==============>
        small_ground_positive = torch.masked_select(small_ground_truth, small_anchor_mark_positive)
        object_num = object_num + len(small_ground_positive)

        if len(small_ground_positive) > 0:

            small_predict_positive = torch.masked_select(samll_bounding_boxes, small_anchor_mark_positive)
            small_box_param = torch.masked_select(small_positive_modulus, small_positive_modulus_mark)

            small_ground_positive = small_ground_positive.view([-1, 5 + self.class_num])
            small_predict_positive = small_predict_positive.view([-1, 5 + self.class_num])
            small_box_param = small_box_param.view([-1, 6])

            for ground_index in range(len(small_ground_positive)):
                ground_box = small_box_param[ground_index][1:5]
                grid_x = int((ground_box[0] + ground_box[2]) / 2 / self.small_downsmaple)
                grid_y = int((ground_box[1] + ground_box[3]) / 2 / self.small_downsmaple)
                anchor_index = small_box_param[ground_index][5].int().item()
                anchor_width, anchor_height = self.anchors_size[anchor_index]
                predict_center_x = (grid_x + small_predict_positive[ground_index][0].item()) * self.small_downsmaple
                predict_center_y = (grid_y + small_predict_positive[ground_index][1].item()) * self.small_downsmaple
                predict_width = anchor_width * math.pow(math.e, small_predict_positive[ground_index][2].item())
                predict_height = anchor_height * math.pow(math.e, small_predict_positive[ground_index][3].item())
                predict_box = [round(predict_center_x - predict_width / 2),
                               round(predict_center_y - predict_height / 2),
                               round(predict_center_x + predict_width - predict_width / 2),
                               round(predict_center_y + predict_height - predict_height / 2)]
                iou_sum = iou_sum + self.iou(predict_box, ground_box)
                #print("iou:{}".format(self.iou(predict_box, ground_box)))
            # positive samples
            coord = self.l_coord * (torch.pow(small_ground_positive[:, 0:2] - small_predict_positive[:, 0:2], 2).sum() / batch_size + \
                    (torch.pow(small_ground_positive[:, 2] - small_predict_positive[:, 2], 2) * small_box_param[:,0]).sum() / batch_size + \
                    (torch.pow(small_ground_positive[:, 3] - small_predict_positive[:, 3], 2) * small_box_param[:, 0]).sum() / batch_size)

            loss = loss + coord
            loss_coord = loss_coord + coord.item()

            confidence = torch.pow(small_ground_positive[:, 4] - small_predict_positive[:, 4], 2).sum() / batch_size
            loss = loss + confidence
            loss_confidence = loss_confidence + confidence.item()

            #small_predict_classes = torch.clamp(small_predict_positive[:, 5:].clone(), min=1e-5, max=1-1e-5)
            classify = bce_loss(small_predict_positive[:, 5:], small_ground_positive[:, 5:])
            loss = loss + classify
            loss_classes = loss_classes + classify.item()

        # negative
        small_ground_negative = torch.masked_select(small_ground_truth, small_anchor_mark_negative)
        if len(small_ground_negative) > 0:
            small_predict_negative = torch.masked_select(samll_bounding_boxes, small_anchor_mark_negative)

            confidence = self.l_noobj * torch.pow(small_ground_negative - small_predict_negative, 2).sum() / batch_size
            loss = loss + confidence
            loss_confidence = loss_confidence + confidence.item()

        #print("loss-1:{} coord:{} conf:{} class:{}".format(loss, coord, confidence, classify))

        #<================middle loss==============>
        middle_ground_positive = torch.masked_select(middle_ground_truth, middle_anchor_mark_positive)
        object_num = object_num + len(middle_ground_positive)
        if len(middle_ground_positive) > 0:
            middle_predict_positive = torch.masked_select(middle_bounding_boxes, middle_anchor_mark_positive)
            middle_box_param = torch.masked_select(middle_positive_modulus, middle_positive_modulus_mark)

            middle_ground_positive = middle_ground_positive.view([-1, 5 + self.class_num])
            middle_predict_positive = middle_predict_positive.view([-1, 5 + self.class_num])
            middle_box_param = middle_box_param.view([-1, 6])
            # positive samples
            for ground_index in range(len(middle_ground_positive)):
                ground_box = middle_box_param[ground_index][1:5]
                grid_x = int((ground_box[0] + ground_box[2]) / 2 / self.middle_downsmaple)
                grid_y = int((ground_box[1] + ground_box[3]) / 2 / self.middle_downsmaple)
                anchor_index = middle_box_param[ground_index][5].int().item()
                anchor_width, anchor_height = self.anchors_size[anchor_index]
                predict_center_x = (grid_x + middle_predict_positive[ground_index][0].item()) * self.middle_downsmaple
                predict_center_y = (grid_y + middle_predict_positive[ground_index][1].item()) * self.middle_downsmaple
                predict_width = anchor_width * math.pow(math.e, middle_predict_positive[ground_index][2].item())
                predict_height = anchor_height * math.pow(math.e, middle_predict_positive[ground_index][3].item())
                predict_box = [round(predict_center_x - predict_width / 2),
                               round(predict_center_y - predict_height / 2),
                               round(predict_center_x + predict_width - predict_width / 2),
                               round(predict_center_y + predict_height - predict_height / 2)]
                iou_sum = iou_sum + self.iou(predict_box, ground_box)
                #print("iou:{}".format(self.iou(predict_box, ground_box)))

            coord = self.l_coord * (torch.pow(middle_ground_positive[:, 0:2] - middle_predict_positive[:, 0:2], 2).sum() / batch_size + \
                    (torch.pow(middle_ground_positive[:, 2] - middle_predict_positive[:, 2], 2) * middle_box_param[:, 0]).sum() / batch_size + \
                    (torch.pow(middle_ground_positive[:, 3] - middle_predict_positive[:, 3], 2) * middle_box_param[:, 0]).sum() / batch_size)

            loss = loss + coord
            loss_coord = loss_coord + coord.item()

            confidence = torch.pow(middle_ground_positive[:, 4] - middle_predict_positive[:, 4], 2).sum() / batch_size
            loss = loss + confidence
            loss_confidence = loss_confidence + confidence.item()

            #middle_predict_classes = torch.clamp(middle_predict_positive[:, 5:], min=1e-5, max=1 - 1e-5)
            classify = bce_loss(middle_predict_positive[:, 5:], middle_ground_positive[:, 5:])
            loss = loss + classify
            loss_classes = loss_classes + classify.item()

        # negative
        middle_ground_negative = torch.masked_select(middle_ground_truth, middle_anchor_mark_negative)
        if len(middle_ground_negative) > 0:
            middle_predict_negative = torch.masked_select(middle_bounding_boxes, middle_anchor_mark_negative)

            confidence = self.l_noobj * torch.pow(middle_ground_negative - middle_predict_negative, 2).sum() / batch_size
            loss = loss + confidence
            loss_confidence = loss_confidence + confidence.item()

        #print("loss-2:{} coord:{} conf:{} class:{}".format(loss, coord, confidence, classify))

        #<=================big loss==============>
        big_ground_positive = torch.masked_select(big_ground_truth, big_anchor_mark_positive)
        big_predict_positive = torch.masked_select(big_bounding_boxes, big_anchor_mark_positive)
        big_box_param = torch.masked_select(big_positive_modulus, big_positive_modulus_mark)

        big_ground_positive = big_ground_positive.view([-1, 5 + self.class_num])
        object_num = object_num + len(big_ground_positive)

        if len(big_ground_positive) > 0:

            big_predict_positive = big_predict_positive.view([-1, 5 + self.class_num])
            big_box_param = big_box_param.view([-1, 6])
            # positive samples
            for ground_index in range(len(big_ground_positive)):
                ground_box = big_box_param[ground_index][1:5]
                grid_x = int((ground_box[0] + ground_box[2]) / 2 / self.big_downsmaple)
                grid_y = int((ground_box[1] + ground_box[3]) / 2 / self.big_downsmaple)
                anchor_index = big_box_param[ground_index][5].int().item()
                anchor_width, anchor_height = self.anchors_size[anchor_index]
                predict_center_x = (grid_x + big_predict_positive[ground_index][0].item()) * self.big_downsmaple
                predict_center_y = (grid_y + big_predict_positive[ground_index][1].item()) * self.big_downsmaple
                predict_width = anchor_width * math.pow(math.e, big_predict_positive[ground_index][2].item())
                predict_height = anchor_height * math.pow(math.e, big_predict_positive[ground_index][3].item())
                predict_box = [round(predict_center_x - predict_width / 2),
                               round(predict_center_y - predict_height / 2),
                               round(predict_center_x + predict_width - predict_width / 2),
                               round(predict_center_y + predict_height - predict_height / 2)]
                iou_sum = iou_sum + self.iou(predict_box, ground_box)
                #print("iou:{}".format(self.iou(predict_box, ground_box)))

            coord = self.l_coord * (torch.pow(big_ground_positive[:, 0:2] - big_predict_positive[:, 0:2], 2).sum() / batch_size + \
                    (torch.pow(big_ground_positive[:, 2] - big_predict_positive[:, 2], 2) * big_box_param[:,0]).sum() / batch_size + \
                    (torch.pow(big_ground_positive[:, 3] - big_predict_positive[:, 3], 2) * big_box_param[:,0]).sum() / batch_size)
            loss = loss + coord
            loss_coord = loss_coord + coord.item()

            confidence = torch.pow(big_ground_positive[:, 4] - big_predict_positive[:, 4], 2).sum() / batch_size
            loss = loss + confidence
            loss_confidence = loss_confidence + confidence.item()

            #big_predict_classes = torch.clamp(big_predict_positive[:, 5:], min=1e-7, max=1 - 1e-7)
            classify = bce_loss(big_predict_positive[:, 5:], big_ground_positive[:, 5:])
            loss = loss + classify
            loss_classes = loss_classes + classify.item()

        # negative
        big_ground_negative = torch.masked_select(big_ground_truth, big_anchor_mark_negative)
        if len(big_ground_negative) > 0:
            big_predict_negative = torch.masked_select(big_bounding_boxes, big_anchor_mark_negative)

            confidence = self.l_noobj * torch.pow(big_ground_negative - big_predict_negative, 2).sum() / batch_size
            loss = loss + confidence
            loss_confidence = loss_confidence + confidence.item()

        #print("loss-3:{} coord:{} conf:{} class:{}".format(loss, coord, confidence, classify))
        #time_end = time.time()
        #print('loss_middle:totally cost:{} loss:{}'.format(time_end - time_start, loss))

        #print("iou:{} num:{}".format(iou_sum, object_num))
        return loss, loss_coord, loss_confidence, loss_classes, iou_sum.item(), object_num

    def setImgSize(self, img_size, anchors_size):
        self.img_size = img_size
        self.anchors_size = anchors_size

'''
    def forward(self, samll_bounding_boxes, middle_bounding_boxes, big_bounding_boxes, small_anchor_mark, small_ground_truth, middle_anchor_mark, middle_ground_truth, big_anchor_mark, big_ground_truth):  # 输入是 S * S * ( 2 * B + Classes)
        # 定义三个计算损失的变量 正样本定位损失 样本置信度损失 样本类别损失
        loss = 0
        loss_coord = 0
        loss_confidence = 0
        loss_classes = 0
        iou_sum = 0
        object_num = 0
        mse_loss = nn.MSELoss()
        #bceLoss = nn.BCELoss()
        positives_num = 0
        negatives_num = 0
        bce_loss = nn.BCEWithLogitsLoss()

        small_grid_feature_size = round(self.img_size / self.small_downsmaple)
        middle_grid_feature_size = round(self.img_size / self.middle_downsmaple)
        big_grid_feature_size = round(self.img_size / self.big_downsmaple)

        time_start = time.time()

        # ground_size, batch_size, width, height, 3个anchor
        #small_ground_truth = small_ground_truth.permute(4, 0, 1, 2, 3)
        #samll_bounding_boxes = samll_bounding_boxes.permute(4, 0, 1, 2, 3)
        small_ground = torch.masked_select(small_ground_truth, small_anchor_mark)
        small_predict = torch.masked_select(samll_bounding_boxes, small_anchor_mark)

        small_ground = small_ground.view([-1, 5 + self.class_num])
        small_predict = small_predict.view([-1, 5 + self.class_num])

        loss = loss + mse_loss(small_ground[:4], small_predict[:4])
        loss = loss + bce_loss(small_ground[4:], small_predict[4:])

        middle_ground = torch.masked_select(middle_ground_truth, middle_anchor_mark)
        middle_predict = torch.masked_select(middle_bounding_boxes, middle_anchor_mark)

        middle_ground = middle_ground.view([-1, 5 + self.class_num])
        middle_predict = middle_predict.view([-1, 5 + self.class_num])

        loss = loss + mse_loss(middle_ground[:4], middle_predict[:4])
        loss = loss + bce_loss(middle_ground[4:], middle_predict[4:])

        big_ground = torch.masked_select(big_ground_truth, big_anchor_mark)
        big_predict = torch.masked_select(big_bounding_boxes, big_anchor_mark)

        big_ground = big_ground.view([-1, 5 + self.class_num])
        big_predict = big_predict.view([-1, 5 + self.class_num])

        loss = loss + mse_loss(big_ground[:4], big_predict[:4])
        loss = loss + bce_loss(big_ground[4:], big_predict[4:])

        time_end = time.time()
        print('loss_middle:totally cost', time_end - time_start)

        return loss, loss_coord, loss_confidence, loss_classes, iou_sum, object_num
'''

