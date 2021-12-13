#------step0: common defination------
import torch
from datetime import datetime

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

class_num = 20
input_size_index = 9
img_sizes = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
anchor_boxes = [[21, 29], [35, 80], [61, 45],  [68, 143], [124, 95], [130, 229], [226, 339], [298, 174], [452, 384]]
weight_file = "../Train/weights/YOLO_V3_200.pth"
#------step1: model------
from Train.YOLO_V3_Model import YOLO_V3
from model import set_freeze_by_idxs, unfreeze_by_idxs
YOLO_V3 = YOLO_V3(class_num=class_num)
YOLO_V3.load_state_dict(torch.load(weight_file, map_location=torch.device("cpu"))["model"])
YOLO_V3 = YOLO_V3.to(device=device)
YOLO_V3.eval()

#------step:2 class_name_dict------
class_file_name = "../DataSet/VOC2007+2012/class.data"
class_index_name = {}
class_index = 0
with open(class_file_name, 'r') as f:
    for line in f:
        line = line.replace('\n', '')
        class_index_name[class_index] = line  # 根据类别名制作索引
        class_index = class_index + 1

#------step:4 NMS------

def iou(box_one, box_two):
    LX = max(box_one[0], box_two[0])
    LY = max(box_one[1], box_two[1])
    RX = min(box_one[2], box_two[2])
    RY = min(box_one[3], box_two[3])
    if LX >= RX or LY >= RY:
        return 0
    return (RX - LX) * (RY - LY) / ((box_one[2]-box_one[0]) * (box_one[3] - box_one[1]) + (box_two[2]-box_two[0]) * (box_two[3] - box_two[1]))

import math
import numpy as np
def NMS_MultiSacle(small_bounding_boxes, middle_bounding_boxes, big_bounding_boxes, input_size, anchor_boxes, small_downsample = 8, middle_downsample = 16, big_downsample = 32, confidence_threshold=0.8, iou_threshold=0.5):
    predict_boxes = []
    nms_boxes = []

    # batch_size * height * witdh * 3 * (5 + class_num)
    small_bounding_boxes = small_bounding_boxes.cpu().detach().numpy()
    middle_bounding_boxes = middle_bounding_boxes.cpu().detach().numpy()
    big_bounding_boxes = big_bounding_boxes.cpu().detach().numpy()

    small_grids_num = input_size // small_downsample
    middle_grids_num = input_size // middle_downsample
    big_grids_num = input_size // big_downsample

    for batch_data in small_bounding_boxes:
        for row_index in range(small_grids_num):
            for col_index in range(small_grids_num):
                small_gird_predict = batch_data[row_index][col_index]
                max_confidence_index = np.argmax(small_gird_predict[::, 4])
                #print(small_gird_predict)
                #print(max_confidence_index)
                small_predict = small_gird_predict[max_confidence_index]
                confidence = small_predict[4]
                if confidence < confidence_threshold:
                    continue
                center_x = (col_index + small_predict[0]) * small_downsample
                center_y = (row_index + small_predict[1]) * small_downsample
                width = anchor_boxes[max_confidence_index][0] * math.pow(math.e, small_predict[2])
                height = anchor_boxes[max_confidence_index][1] * math.pow(math.e, small_predict[3])
                xmin = max(0, round(center_x - width / 2))
                ymin = max(0, round(center_y - height / 2))
                xmax = min(input_size - 1, round(center_x + width / 2))
                ymax = min(input_size - 1, round(center_y + height / 2))
                class_index = np.argmax(small_predict[5:])
                predict_box = [xmin, ymin, xmax, ymax, confidence, class_index]
                predict_boxes.append(predict_box)

    for batch_data in middle_bounding_boxes:
        for row_index in range(middle_grids_num):
            for col_index in range(middle_grids_num):
                middle_gird_predict = batch_data[row_index][col_index]
                max_confidence_index = np.argmax(middle_gird_predict[::, 4])
                middle_predict = middle_gird_predict[max_confidence_index]
                confidence = middle_predict[4]
                if confidence < confidence_threshold:
                    continue
                center_x = (col_index + middle_predict[0]) * middle_downsample
                center_y = (row_index + middle_predict[1]) * middle_downsample
                width = anchor_boxes[3 + max_confidence_index][0] * math.pow(math.e, middle_predict[2])
                height = anchor_boxes[3 + max_confidence_index][1] * math.pow(math.e, middle_predict[3])
                xmin = max(0, round(center_x - width / 2))
                ymin = max(0, round(center_y - height / 2))
                xmax = min(input_size - 1, round(center_x + width / 2))
                ymax = min(input_size - 1, round(center_y + height / 2))
                class_index = np.argmax(middle_predict[5:])
                predict_box = [xmin, ymin, xmax, ymax, confidence, class_index]
                predict_boxes.append(predict_box)

    for batch_data in big_bounding_boxes:
        for row_index in range(big_grids_num):
            for col_index in range(big_grids_num):
                big_gird_predict = batch_data[row_index][col_index]
                max_confidence_index = np.argmax(big_gird_predict[::, 4])
                big_predict = big_gird_predict[max_confidence_index]
                confidence = round(big_predict[4], 2)
                if confidence < confidence_threshold:
                    continue
                center_x = (col_index + big_predict[0]) * middle_downsample
                center_y = (row_index + big_predict[1]) * middle_downsample
                width = anchor_boxes[3 + max_confidence_index][0] * math.pow(math.e, big_predict[2])
                height = anchor_boxes[3 + max_confidence_index][1] * math.pow(math.e, big_predict[3])
                xmin = max(0, round(center_x - width / 2))
                ymin = max(0, round(center_y - height / 2))
                xmax = min(input_size - 1, round(center_x + width / 2))
                ymax = min(input_size - 1, round(center_y + height / 2))
                class_index = np.argmax(big_predict[5:])
                predict_box = [xmin, ymin, xmax, ymax, confidence, class_index]
                predict_boxes.append(predict_box)

    while len(predict_boxes) != 0:
        predict_boxes.sort(key=lambda box: box[4])
        assured_box = predict_boxes[0]
        temp = []
        nms_boxes.append(assured_box)
        i = 1
        while i < len(predict_boxes):
            if iou(assured_box, predict_boxes[i]) <= iou_threshold:
                temp.append(predict_boxes[i])
            i = i + 1
        predict_boxes = temp

    return nms_boxes

def NMS(bounding_boxes,S=7,B=2,img_size=448,confidence_threshold=0.9,iou_threshold=0.5):

    predict_boxes = []
    nms_boxes = []
    grid_size = img_size / S
    for batch in range(len(bounding_boxes)):
        for i in range(S):
            for j in range(S):
                gridX = grid_size * j
                gridY = grid_size * i
                if bounding_boxes[batch][i][j][4] < bounding_boxes[batch][i][j][9]:
                    bounding_box = bounding_boxes[batch][i][j][5:10]
                else:
                    bounding_box = bounding_boxes[batch][i][j][0:5]
                class_possible = (bounding_boxes[batch][i][j][10:])
                bounding_box.extend(class_possible)
                if bounding_box[4] < confidence_threshold:
                    continue
                centerX = (int)(gridX + bounding_box[0] * grid_size)
                centerY = (int)(gridY + bounding_box[1] * grid_size)
                width = (int)(bounding_box[2] * img_size)
                height = (int)(bounding_box[3] * img_size)
                bounding_box[0] = max(0, (int)(centerX - width / 2))
                bounding_box[1] = max(0, (int)(centerY - height / 2))
                bounding_box[2] = min(img_size - 1, (int)(centerX + width / 2))
                bounding_box[3] = min(img_size - 1, (int)(centerY + height / 2))
                predict_boxes.append(bounding_box)

        while len(predict_boxes) != 0:
            predict_boxes.sort(key=lambda box:box[4])
            assured_box = predict_boxes[0]
            temp = []
            classIndex = np.argmax(assured_box[5:])
            #print("类别索引:{}".format(classIndex))
            assured_box[4] = assured_box[4] * assured_box[5 + classIndex] #修正置信度为 物体分类准确度 × 含有物体的置信度
            assured_box[5] = classIndex
            nms_boxes.append(assured_box)
            i = 1
            while i < len(predict_boxes):
                if iou(assured_box,predict_boxes[i]) <= iou_threshold:
                    temp.append(predict_boxes[i])
                i = i + 1
            predict_boxes = temp

        return nms_boxes

#------step:5 detection ------
import cv2
from utils import image
import torchvision.transforms as transforms
img_file_name = "../DataSet/VOC2007+2012/Train/JPEGImages/000002.jpg"
input_size = img_sizes[input_size_index]

transform = transforms.Compose([
    transforms.ToTensor(), # height * width * channel -> channel * height * width
    transforms.Normalize(mean=(0.408, 0.448, 0.471), std=(0.242, 0.239, 0.234))
])
img = cv2.imread(img_file_name)
img = image.resize_image_without_annotation(img, input_size, input_size)
train_data = transform(img)
train_data = train_data.unsqueeze(0)
train_data = train_data.to(device=device)

with torch.no_grad():
    small_bounding_boxes, middle_bounding_boxes, big_bounding_boxes = YOLO_V3(train_data)
NMS_boxes = NMS_MultiSacle(small_bounding_boxes, middle_bounding_boxes, big_bounding_boxes, input_size, anchor_boxes)

for box in NMS_boxes:
    print(box)
    img = cv2.rectangle(img, (box[0],box[1]),(box[2],box[3]),(0,255,0),1)
    confidence = round(box[4],2)
    img = cv2.putText(img, "{}-{}".format(class_index_name[box[5]],confidence),(box[0],box[1]),cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,255),1)
    print("class_name:{} confidence:{}".format(class_index_name[box[5]],confidence))

cv2.imshow("img_detection",img)
cv2.waitKey()
cv2.destroyAllWindows()


