from torch.utils.data import Dataset
import os
import cv2
import xml.etree.ElementTree as ET
import torch
import torchvision.transforms as transforms
import math
import numpy as np
import random
from utils import image

class VOCDataSet(Dataset):
    def __init__(self, imgs_path="../DataSet/VOC2007+2012/Train/JPEGImages", annotations_path="../DataSet/VOC2007+2012/Train/Annotations", classes_file="../DataSet/VOC2007+2012/class.data", is_train=True, class_num=20, iou_threshold=0.3, label_smooth_value=0.05, small_downsample=8, middle_downsample=16, big_downsample=32): #input_size:输入图像的尺度
        self.small_downsmaple = small_downsample
        self.middle_downsmaple = middle_downsample
        self.big_downsmaple = big_downsample
        self.label_smooth_value = label_smooth_value
        self.class_num = class_num
        self.iou_threshold = iou_threshold
        self.imgs_name = os.listdir(imgs_path)
        self.is_train = is_train

        self.transform_common = transforms.Compose([
            transforms.ToTensor(),  # height * width * channel -> channel * height * width
            transforms.Normalize(mean=(0.408, 0.448, 0.471), std=(0.242, 0.239, 0.234))  # 归一化后.不容易产生梯度爆炸的问题
        ])

        self.imgs_path = imgs_path
        self.annotations_path = annotations_path
        self.class_dict = {}
        class_index = 0
        with open(classes_file, 'r') as file:
            for class_name in file:
                class_name = class_name.replace('\n', '')
                self.class_dict[class_name] = class_index  # 根据类别名制作索引
                class_index = class_index + 1

    def __getitem__(self, item):
        transform_seed = random.randint(0, 4)
        img_path = os.path.join(self.imgs_path, self.imgs_name[item])
        annotation_path = os.path.join(self.annotations_path, self.imgs_name[item].replace(".jpg", ".xml"))
        img = cv2.imread(img_path)
        tree = ET.parse(annotation_path)
        annotation_xml = tree.getroot()

        if self.is_train or transform_seed == 0:  # 原图
            img, coords = image.resize_image(img, self.input_size, self.input_size, annotation_xml, self.class_dict)
            #img = self.transform_common(img)
            #small_anchor_mark, small_ground_truth, middle_anchor_mark, middle_ground_truth, big_anchor_mark, big_ground_truth = self.getGroundTruth(coords)

        elif transform_seed == 1:  # 缩放+中心裁剪
            img, annotation_xml = image.center_crop(img, annotation_xml)
            img, coords = image.resize_image(img, self.input_size, self.input_size, annotation_xml, self.class_dict)
            #img = self.transform_common(img)
            #small_anchor_mark, small_ground_truth, middle_anchor_mark, middle_ground_truth, big_anchor_mark, big_ground_truth = self.getGroundTruth(coords)

        elif transform_seed == 2:  # 平移
            img, annotation_xml = image.transplant(img, annotation_xml)
            img, coords = image.resize_image(img, self.input_size, self.input_size, annotation_xml, self.class_dict)
            #img = self.transform_common(img)
            #small_anchor_mark, small_ground_truth, middle_anchor_mark, middle_ground_truth, big_anchor_mark, big_ground_truth = self.getGroundTruth(coords)

        elif transform_seed == 3:  # 明度调整 YOLO在论文中称曝光度为明度
            img, coords = image.resize_image(img, self.input_size, self.input_size, annotation_xml, self.class_dict)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            H, S, V = cv2.split(img)
            cv2.merge([np.uint8(H), np.uint8(S), np.uint8(V * 1.5)], dst=img)
            cv2.cvtColor(src=img, dst=img, code=cv2.COLOR_HSV2BGR)
            #img = self.transform_common(img)
            #small_anchor_mark, small_ground_truth, middle_anchor_mark, middle_ground_truth, big_anchor_mark, big_ground_truth = self.getGroundTruth(coords)

        else:  # 饱和度调整
            img, coords = image.resize_image(img, self.input_size, self.input_size, annotation_xml, self.class_dict)
            H, S, V = cv2.split(img)
            cv2.merge([np.uint8(H), np.uint8(S * 1.5), np.uint8(V)], dst=img)
            cv2.cvtColor(src=img, dst=img, code=cv2.COLOR_HSV2BGR)


        small_ground_truth, small_positive_modulus, small_anchor_mark_positive, small_anchor_mark_negative, small_positive_modulus_mark, middle_ground_truth, middle_positive_modulus, middle_anchor_mark_positive, middle_anchor_mark_negative, middle_positive_modulus_mark, big_ground_truth, big_positive_modulus, big_anchor_mark_positive, big_anchor_mark_negative, big_positive_modulus_mark = self.getGroundTruth(coords)
        img = self.transform_common(img)
        # 通道变化方法: img = img[:, :, ::-1]

        return img, small_ground_truth, small_positive_modulus, small_anchor_mark_positive, small_anchor_mark_negative, small_positive_modulus_mark, middle_ground_truth, middle_positive_modulus, middle_anchor_mark_positive, middle_anchor_mark_negative, middle_positive_modulus_mark, big_ground_truth, big_positive_modulus, big_anchor_mark_positive, big_anchor_mark_negative, big_positive_modulus_mark


    def __len__(self):
        return len(self.imgs_name)

    def setInputSize(self, input_size, anchors_size):
        self.input_size = input_size
        self.anchors_size = anchors_size

    def anchor_ground_IoU(self, anchor, ground):# anchor:width height  ground:centerX centerY width height
        # 1.第一种实现方案 将左上角对齐直接算IoU
        # 2.第二种方案 将anchor与ground中心对齐后计算IoU
        # 注意：这两种方案实现起来得到最终的IoU值是一样的
        interArea = min(anchor[0], ground[0]) * min(anchor[1], ground[1])
        Area_anchor = anchor[0] * anchor[1]
        Area_ground = ground[0] * ground[1]
        return interArea / (Area_anchor + Area_ground - interArea)


    def read_img(self, img_path, annotation_path):
        img = cv2.imread(img_path)
        tree = ET.parse(annotation_path)
        annotation_xml = tree.getroot()
        img, coords = image.resize_image(img, self.input_size, self.input_size, annotation_xml, self.class_dict)
        small_ground_truth, small_positive_modulus, small_anchor_mark_positive, small_anchor_mark_negative, small_positive_modulus_mark, middle_ground_truth, middle_positive_modulus, middle_anchor_mark_positive, middle_anchor_mark_negative, middle_positive_modulus_mark, big_ground_truth, big_positive_modulus, big_anchor_mark_positive, big_anchor_mark_negative, big_positive_modulus_mark = self.getGroundTruth(coords, img)

    def getGroundTruth(self, coords):

        small_feature_size = round(self.input_size / self.small_downsmaple)
        middle_feature_size = round(self.input_size / self.middle_downsmaple)
        big_feature_size = round(self.input_size / self.big_downsmaple)

        small_ground_truth = np.zeros([small_feature_size, small_feature_size, 3, 5 + self.class_num])  # YOLO V3中正样本的置信度为1, 此处将置信度替换为大小样本宽高损失的权衡系数
        middle_ground_truth = np.zeros([middle_feature_size, middle_feature_size, 3, 5 + self.class_num])
        big_ground_truth = np.zeros([big_feature_size, big_feature_size, 3, 5 + self.class_num])

        # positive mask
        small_anchor_mark_positive = np.zeros([small_feature_size, small_feature_size, 3, 5 + self.class_num])
        middle_anchor_mark_positive = np.zeros([middle_feature_size, middle_feature_size, 3, 5 + self.class_num])
        big_anchor_mark_positive = np.zeros([big_feature_size, big_feature_size, 3, 5 + self.class_num])

        # negative mask
        small_anchor_mark_negative = np.zeros([small_feature_size, small_feature_size, 3, 5 + self.class_num])
        middle_anchor_mark_negative = np.zeros([middle_feature_size, middle_feature_size, 3, 5 + self.class_num])
        big_anchor_mark_negative = np.zeros([big_feature_size, big_feature_size, 3, 5 + self.class_num])

        # positive modulus
        small_positive_modulus = np.zeros([small_feature_size, small_feature_size, 3, 6])
        middle_positive_modulus = np.zeros([middle_feature_size, middle_feature_size, 3, 6])
        big_positive_modulus = np.zeros([big_feature_size, big_feature_size, 3, 6])

        # modulus mask
        small_positive_modulus_mark = np.zeros([small_feature_size, small_feature_size, 3, 6])
        middle_positive_modulus_mark = np.zeros([middle_feature_size, middle_feature_size, 3, 6])
        big_positive_modulus_mark = np.zeros([big_feature_size, big_feature_size, 3, 6])

        #扣出置信度 首先默认所有的都是负样本
        small_anchor_mark_negative[:, :, :, 4] = 1
        middle_anchor_mark_negative[:, :, :, 4] = 1
        big_anchor_mark_negative[:, :, :, 4] = 1

        for coord in coords:
            # bounding box归一化
            [xmin, ymin, xmax, ymax, class_index] = coord

            ground_width = (xmax - xmin)
            ground_height = (ymax - ymin)

            box_width = ground_width * self.input_size
            box_height = ground_height * self.input_size

            centerX = (xmin + xmax) / 2
            centerY = (ymin + ymax) / 2

            # 计算当前中心点分别落于3个特征尺度下的哪个grid内
            small_indexRow = (int)(centerY * small_feature_size)
            small_indexCol = (int)(centerX * small_feature_size)

            '''
            cx = round(centerX * small_feature_size * self.small_downsmaple)
            cy = round(centerY * small_feature_size * self.small_downsmaple)
            sx = max(0, round(cx - box_width / 2))
            sy = max(0, round(cy - box_height / 2))
            ox = min(self.input_size - 1, round(cx + box_width / 2))
            oy = min(self.input_size - 1, round(cy + box_height / 2))
            print("cx:{} cy:{} width:{} height:{} sx:{} sy:{} ox:{} oy:{}".format(cx, cy, box_width, box_height, sx, sy, ox, oy))
            img = cv2.rectangle(img, (sx, sy), (ox, oy), (0, 255, 0), 1)
            '''

            middle_indexRow = (int)(centerY * middle_feature_size)
            middle_indexCol = (int)(centerX * middle_feature_size)

            '''
            cx = round(centerX * middle_feature_size * self.middle_downsmaple)
            cy = round(centerY * middle_feature_size * self.middle_downsmaple)
            sx = max(0, round(cx - box_width / 2))
            sy = max(0, round(cy - box_height / 2))
            ox = min(self.input_size - 1, round(cx + box_width / 2))
            oy = min(self.input_size - 1, round(cy + box_height / 2))
            #print("cx:{} cy:{} width:{} height:{} sx:{} sy:{} ox:{} oy:{}".format(cx, cy, box_width, box_height, sx, sy, ox, oy))
            '''

            big_indexRow = (int)(centerY * big_feature_size)
            big_indexCol = (int)(centerX * big_feature_size)

            '''
            cx = round(centerX * big_feature_size * self.big_downsmaple)
            cy = round(centerY * big_feature_size * self.big_downsmaple)
            sx = max(0, round(cx - box_width / 2))
            sy = max(0, round(cy - box_height / 2))
            ox = min(self.input_size - 1, round(cx + box_width / 2))
            oy = min(self.input_size - 1, round(cy + box_height / 2))
            print("cx:{} cy:{} width:{} height:{} sx:{} sy:{} ox:{} oy:{}".format(cx, cy, box_width, box_height, sx, sy, ox, oy))
            '''
            max_iou = 0
            max_iou_index = -1
            for anchor_index in range(len(self.anchors_size)):
                anchor_size = self.anchors_size[anchor_index]
                iou = self.anchor_ground_IoU(anchor_size, [box_width, box_height])
                #print("iou-index:{} iou:{}".format(anchor_index, iou))
                if iou > self.iou_threshold:

                    if anchor_index < 3 and small_anchor_mark_positive[small_indexRow][small_indexCol][anchor_index][0] == 0:
                        small_anchor_mark_negative[small_indexRow][small_indexCol][anchor_index][4] = 0
                        #small_anchor_mark_negative[small_indexRow][small_indexCol][anchor_index] = np.zeros(5 + self.class_num) # ingore sample

                    elif anchor_index >= 3 and anchor_index < 6 and middle_anchor_mark_positive[middle_indexRow][middle_indexCol][anchor_index - 3][0] == 0:
                        middle_anchor_mark_negative[middle_indexRow][middle_indexCol][anchor_index - 3][4] = 0
                        #middle_anchor_mark_negative[middle_indexRow][middle_indexCol][anchor_index - 3] = np.zeros(5 + self.class_num) # confidence_mark

                    elif anchor_index >= 6 and anchor_index < 9 and big_anchor_mark_positive[big_indexRow][big_indexCol][anchor_index - 6][0] == 0:
                        big_anchor_mark_negative[big_indexRow][big_indexCol][anchor_index - 6][4] = 0
                        #big_anchor_mark_negative[big_indexRow][big_indexCol][anchor_index - 6] = np.zeros(5 + self.class_num) # confidence_mark

                if iou > max_iou:
                    max_iou = iou
                    max_iou_index = anchor_index

            scale_width = math.log(box_width / self.anchors_size[max_iou_index][0])
            scale_height = math.log(box_height / self.anchors_size[max_iou_index][1])
            # 大小物体损失值平衡系数
            scale_adjust_modulu = 2 - ground_width * ground_height

            #test------
            '''
            cx = round(centerX * small_feature_size * self.small_downsmaple)
            cy = round(centerY * small_feature_size * self.small_downsmaple)
            box_width = math.pow(math.e, scale_width) * self.anchors_size[max_iou_index][0]
            box_height = math.pow(math.e, scale_height) * self.anchors_size[max_iou_index][1]
            sx = max(0, round(cx - box_width / 2))
            sy = max(0, round(cy - box_height / 2))
            ox = min(self.input_size - 1, round(cx + box_width / 2))
            oy = min(self.input_size - 1, round(cy + box_height / 2))
            print("ground:{} {} {} {} ---- point:{} {} {} {} class:{}".format(cx, cy, box_width, box_height, sx, sy, ox, oy, class_index))
            img = cv2.rectangle(img, (sx, sy), (ox, oy), (0, 255, 0), 1)
            cv2.imshow("return", img)
            cv2.waitKey(1000)
            '''

            # 分类标签 label_smooth
            '''
            # 转化为one_hot编码，将物体的类别设置为1，其他为0
            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            # 对one_hot编码做平滑处理          
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
            
            # 计算中心点坐标(x,y) = ((x_max, y_max) + (x_min, y_min)) * 0.5
            # 计算宽高(w,h) = (x_max, y_max) - (x_min, y_min)
            # 拼接成一个数组(x, y, w, h)
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            
            '''
            class_one_hot = [0 for _ in range(self.class_num)]
            class_list = [self.label_smooth_value / (self.class_num - 1) for i in range(self.class_num)]
            class_list[class_index] = 1 - self.label_smooth_value

            # 定位数据预设
            ground_box = [0, 0, scale_width, scale_height, 1]
            ground_box.extend(class_list)
            modulus = [scale_adjust_modulu, self.input_size * xmin, self.input_size * ymin, self.input_size * xmax, self.input_size * ymax, max_iou_index]

            #print(ground_box)
            #print(max_iou_index)

            if max_iou_index < 3:
                #cnt = cnt + 1
                # 已经使用过的需要标记
                small_anchor_mark_positive[small_indexRow][small_indexCol][max_iou_index] = np.ones([5 + self.class_num])
                # 定位数据
                center_y = centerY * small_feature_size
                ground_box[1] = center_y - small_indexRow
                #print("small center_y:{} ground_box[1]:{}".format(center_y, ground_box[1]))
                center_x = centerX * small_feature_size
                ground_box[0] = center_x - small_indexCol
                #print("small center_x:{} ground_box[0]:{}".format(center_x, ground_box[0]))

                # test------------
                '''
                cy = round((ground_box[1] + small_indexRow) * self.small_downsmaple)
                cx = round((ground_box[0] + small_indexCol) * self.small_downsmaple)
                box_width = math.pow(math.e, ground_box[2]) * self.anchors_size[max_iou_index][0]
                box_height = math.pow(math.e, ground_box[3]) * self.anchors_size[max_iou_index][1]
                sx = max(0, round(cx - box_width / 2))
                sy = max(0, round(cy - box_height / 2))
                ox = min(self.input_size - 1, round(cx + box_width / 2))
                oy = min(self.input_size - 1, round(cy + box_height / 2))
                img = cv2.rectangle(img, (sx, sy), (ox, oy), (0, 255, 0), 1)

                positive_num = positive_num + 1
                '''

                #modulus[5] = max_iou_index
                small_ground_truth[small_indexRow][small_indexCol][max_iou_index] = np.array(ground_box)
                small_positive_modulus[small_indexRow][small_indexCol][max_iou_index] = np.array(modulus)
                small_positive_modulus_mark[small_indexRow][small_indexCol][max_iou_index] = np.ones([6])
                small_anchor_mark_negative[small_indexRow][small_indexCol][max_iou_index] = np.zeros([5 + self.class_num])
                #small_anchor_mark_negative[small_indexRow][small_indexCol][max_iou_index][4] = 0
                #print("small assign row:{} col:{}".format(small_indexRow, small_indexCol))

            elif max_iou_index < 6:
                #cnt = cnt + 1
                middle_anchor_mark_positive[middle_indexRow][middle_indexCol][max_iou_index - 3] = np.ones([5 + self.class_num])
                center_y = centerY * middle_feature_size
                ground_box[1] = center_y - middle_indexRow # offsetY
                #print("middle center_y:{} ground_box[1]:{}".format(center_y, ground_box[1]))
                center_x = centerX * middle_feature_size
                ground_box[0] = center_x - middle_indexCol # offsetX
                #print("middle center_x:{} ground_box[0]:{}".format(center_x, ground_box[0]))

                # test------------
                '''
                cy = round((ground_box[1] + middle_indexRow) * self.middle_downsmaple)
                cx = round((ground_box[0] + middle_indexCol) * self.middle_downsmaple)
                box_width = math.pow(math.e, ground_box[2]) * self.anchors_size[max_iou_index][0]
                box_height = math.pow(math.e, ground_box[3]) * self.anchors_size[max_iou_index][1]
                sx = max(0, round(cx - box_width / 2))
                sy = max(0, round(cy - box_height / 2))
                ox = min(self.input_size - 1, round(cx + box_width / 2))
                oy = min(self.input_size - 1, round(cy + box_height / 2))
                img = cv2.rectangle(img, (sx, sy), (ox, oy), (0, 255, 0), 1)

                positive_num = positive_num + 1
                '''

                middle_ground_truth[middle_indexRow][middle_indexCol][max_iou_index - 3] = np.array(ground_box)
                middle_positive_modulus[middle_indexRow][middle_indexCol][max_iou_index - 3] = np.array(modulus)
                middle_positive_modulus_mark[middle_indexRow][middle_indexCol][max_iou_index - 3] = np.ones([6])
                middle_anchor_mark_negative[middle_indexRow][middle_indexCol][max_iou_index - 3] = np.zeros([5 + self.class_num])
                #middle_anchor_mark_negative[middle_indexRow][middle_indexCol][max_iou_index - 3][4] = 0
                #print("middle assign row:{} col:{}".format(middle_indexRow, middle_indexCol))

            else:
                #cnt = cnt + 1
                big_anchor_mark_positive[big_indexRow][big_indexCol][max_iou_index - 6] = np.ones([5 + self.class_num])
                center_y = centerY * big_feature_size
                ground_box[1] = center_y - big_indexRow
                #print("big center_y:{} ground_box[1]:{}".format(center_y, ground_box[1]))
                center_x = centerX * big_feature_size
                ground_box[0] = center_x - big_indexCol
                #print("big center_x:{} ground_box[0]:{}".format(center_x, ground_box[0]))

                # test------------
                '''
                cy = round((ground_box[1] + big_indexRow) * self.big_downsmaple)
                cx = round((ground_box[0] + big_indexCol) * self.big_downsmaple)
                box_width = math.pow(math.e, ground_box[2]) * self.anchors_size[max_iou_index][0]
                box_height = math.pow(math.e, ground_box[3]) * self.anchors_size[max_iou_index][1]
                sx = max(0, round(cx - box_width / 2))
                sy = max(0, round(cy - box_height / 2))
                ox = min(self.input_size - 1, round(cx + box_width / 2))
                oy = min(self.input_size - 1, round(cy + box_height / 2))
                img = cv2.rectangle(img, (sx, sy), (ox, oy), (0, 255, 0), 1)

                positive_num = positive_num + 1
                '''

                big_ground_truth[big_indexRow][big_indexCol][max_iou_index - 6] = np.array(ground_box, dtype=np.float)
                big_positive_modulus[big_indexRow][big_indexCol][max_iou_index - 6] = np.array(modulus)
                big_positive_modulus_mark[big_indexRow][big_indexCol][max_iou_index - 6] = np.ones([6])
                big_anchor_mark_negative[big_indexRow][big_indexCol][max_iou_index - 6] = np.zeros([5 + self.class_num])
                #big_anchor_mark_negative[big_indexRow][big_indexCol][max_iou_index - 6][4] = 0
                #print("big assign row:{} col:{}".format(big_indexRow, big_indexCol))
        '''
        for row in range(small_feature_size):
            for col in range(small_feature_size):
                for index in range(3):
                    if small_anchor_mark_positive[row][col][index][0] == 1:#positive sample
                        small_ground = small_ground_truth[row][col][index]
                        offset_x = col + small_ground[0]
                        offset_y = row + small_ground[1]
                        scale_width = small_ground[2]
                        scale_height = small_ground[3]

                        width = math.pow(math.e, scale_width) * self.anchors_size[index][0]
                        height = math.pow(math.e, scale_height) * self.anchors_size[index][1]

                        center_x = offset_x * self.small_downsmaple
                        center_y = offset_y * self.small_downsmaple

                        #print("cx:{} cy:{} w:{} h:{}".format(center_x, center_y, width, height))

                        xmin = max(0, round(center_x - width / 2))
                        ymin = max(0, round(center_y - height / 2))
                        xmax = min(607, round(center_x + width / 2))
                        ymax = min(607, round(center_y + height / 2))

                        print("small row:{} col:{} xmin:{} ymin:{} xmax:{} ymax:{} class_index:{}".format(row, col, xmin, ymin, xmax, ymax, np.argmax(small_ground[5:])))
                        #print("small {} {} {} {}".format(round(center_x - width / 2), round(center_y - height / 2), round(center_x + width / 2), round(center_y + height / 2)))

                        img = cv2.rectangle(img, (xmin,ymin),(xmax,ymax),(0,255,0),1)
                        cv2.imshow("return", img)
                        cv2.waitKey(1000)
                        break

        for row in range(middle_feature_size):
            for col in range(middle_feature_size):
                for index in range(3):
                    if middle_anchor_mark_positive[row][col][index][0] == 1:#positive sample
                        middle_ground = middle_ground_truth[row][col][index]
                        offset_x = col + middle_ground[0]
                        offset_y = row + middle_ground[1]
                        scale_width = middle_ground[2]
                        scale_height = middle_ground[3]

                        width = math.pow(math.e, scale_width) * self.anchors_size[index + 3][0]
                        height = math.pow(math.e, scale_height) * self.anchors_size[index + 3][1]

                        center_x = offset_x * self.middle_downsmaple
                        center_y = offset_y * self.middle_downsmaple

                        #print("cx:{} cy:{} w:{} h:{}".format(center_x, center_y, width, height))

                        xmin = max(0, round(center_x - width / 2))
                        ymin = max(0, round(center_y - height / 2))
                        xmax = min(607, round(center_x + width / 2))
                        ymax = min(607, round(center_y + height / 2))

                        print("middle row:{} col:{} middle xmin:{} ymin:{} xmax:{} ymax:{} class_index:{}".format(row, col, xmin, ymin, xmax, ymax, np.argmax(middle_ground[5:])))
                        #print("middle {} {} {} {}".format(round(center_x - width / 2),round(center_y - height / 2),round(center_x + width / 2),round(center_y + height / 2)))

                        img = cv2.rectangle(img, (xmin,ymin),(xmax,ymax),(0,255,0),1)
                        cv2.imshow("return", img)
                        cv2.waitKey(1000)
                        break

        for row in range(big_feature_size):
            for col in range(big_feature_size):
                for index in range(3):
                    if big_anchor_mark_positive[row][col][index][4] == 1:#positive sample
                        #print("big-sample row:{} col:{} index:{}".format(row, col, index))
                        big_ground = big_ground_truth[row][col][index]
                        #print(big_ground)
                        offset_x = col + big_ground[0]
                        offset_y = row + big_ground[1]
                        scale_width = big_ground[2]
                        scale_height = big_ground[3]

                        width = math.pow(math.e, scale_width) * self.anchors_size[index + 6][0]
                        height = math.pow(math.e, scale_height) * self.anchors_size[index + 6][1]

                        center_x = offset_x * self.big_downsmaple
                        center_y = offset_y * self.big_downsmaple

                        #print("cx:{} cy:{} w:{} h:{}".format(center_x, center_y, width, height))

                        xmin = max(0, round(center_x - width / 2))
                        ymin = max(0, round(center_y - height / 2))
                        xmax = min(607, round(center_x + width / 2))
                        ymax = min(607, round(center_y + height / 2))

                        print("big row:{} col:{} xmin:{} ymin:{} xmax:{} ymax:{} class_index:{}".format(row, col, xmin, ymin, xmax, ymax, np.argmax(big_ground[5:])))
                        #print("big {} {} {} {}".format(round(center_x - width / 2),round(center_y - height / 2),round(center_x + width / 2),round(center_y + height / 2)))

                        img = cv2.rectangle(img, (xmin,ymin),(xmax,ymax),(0,255,0),1)
                        cv2.imshow("return", img)
                        cv2.waitKey(1000)
                        break


        cv2.imshow("return", img)
        cv2.waitKey()
        '''
                        #print("cnt:{} positive:{}".format(cnt, positive_num))
        #small_ground_truth = torch.as_tensor(small_ground_truth)
        #middle_ground_truth = torch.as_tensor(middle_ground_truth)
        #big_ground_truth = torch.as_tensor(big_ground_truth)
        #return torch.Tensor(small_anchor_mark).bool(), small_ground_truth, torch.Tensor(middle_anchor_mark).bool(), middle_ground_truth, torch.Tensor(big_anchor_mark).bool(), big_ground_truth
        return torch.Tensor(small_ground_truth).float(), small_positive_modulus, torch.Tensor(small_anchor_mark_positive).bool(), torch.Tensor(small_anchor_mark_negative).bool(), torch.Tensor(small_positive_modulus_mark).bool(),\
               torch.Tensor(middle_ground_truth).float(), middle_positive_modulus, torch.Tensor(middle_anchor_mark_positive).bool(), torch.Tensor(middle_anchor_mark_negative).bool(), torch.Tensor(middle_positive_modulus_mark).bool(),\
               torch.Tensor(big_ground_truth).float(), big_positive_modulus, torch.Tensor(big_anchor_mark_positive).bool(), torch.Tensor(big_anchor_mark_negative).bool(), torch.Tensor(big_positive_modulus_mark).bool(),

    '''
    def getGroundTruth(self, annotation_path):
        
        small_feature_size = round(self.input_size / self.small_downsmaple)
        middle_feature_size = round(self.input_size / self.middle_downsmaple)
        big_feature_size = round(self.input_size / self.big_downsmaple)

        small_ground_truth = np.zeros([small_feature_size, small_feature_size, 9 + self.class_num]) #YOLO V3中正样本的置信度为1, 此处将置信度替换为大小样本宽高损失的权衡系数
        middle_ground_truth = np.zeros([middle_feature_size, middle_feature_size, 9 + self.class_num])
        big_ground_truth = np.zeros([big_feature_size, big_feature_size, 9 + self.class_num])

        small_anchor_mark = np.zeros([small_feature_size, small_feature_size, 3])
        middle_anchor_mark = np.zeros([middle_feature_size, middle_feature_size, 3])
        big_anchor_mark = np.zeros([big_feature_size, big_feature_size, 3])

        tree = ET.parse(annotation_path)
        annotation_xml = tree.getroot()
        img_width = round((float)(annotation_xml.find("size").find("width").text))
        img_height = round((float)(annotation_xml.find("size").find("height").text))
        objects_xml = annotation_xml.findall("object")

        for object_xml in objects_xml:
            class_name = object_xml.find("name").text
            if class_name not in self.ClassNameToClassIndex:  # 不属于我们规定的类
                continue
            bnd_xml = object_xml.find("bndbox")
            # bounding box归一化
            xmin = (float)(bnd_xml.find("xmin").text) / img_width
            ymin = (float)(bnd_xml.find("ymin").text) / img_height
            xmax = (float)(bnd_xml.find("xmax").text) / img_width
            ymax = (float)(bnd_xml.find("ymax").text) / img_height

            ground_width = (xmax - xmin)
            ground_height = (ymax - ymin)

            box_width = ground_width * self.input_size
            box_height = ground_height * self.input_size

            centerX = (xmin + xmax) / 2
            centerY = (ymin + ymax) / 2

            # 计算当前中心点分别落于3个特征尺度下的哪个grid内
            small_indexRow = (int)(centerY * small_feature_size)
            small_indexCol = (int)(centerX * small_feature_size)

            middle_indexRow = (int)(centerY * middle_feature_size)
            middle_indexCol = (int)(centerX * middle_feature_size)

            big_indexRow = (int)(centerY * big_feature_size)
            big_indexCol = (int)(centerX * big_feature_size)

            max_iou = 0
            max_iou_index = -1
            for anchor_index in range(len(self.anchors_size)):
                anchor_size = self.anchors_size[anchor_index]
                iou = self.anchor_ground_IoU(anchor_size, [box_width, box_height])
                if iou > self.iou_threshold:
                    if anchor_index < 3 and small_anchor_mark[small_indexRow][small_indexCol][anchor_index] != 1:
                        small_anchor_mark[small_indexRow][small_indexCol][anchor_index] = -1

                    elif anchor_index < 6 and middle_anchor_mark[middle_indexRow][middle_indexCol][anchor_index % 3] != 1:
                        middle_anchor_mark[middle_indexRow][middle_indexCol][anchor_index % 3] = -1

                    elif big_anchor_mark[big_indexRow][big_indexCol][anchor_index % 3] != 1:
                        big_anchor_mark[big_indexRow][big_indexCol][anchor_index % 3] = -1

                if iou > max_iou:
                    max_iou = iou
                    max_iou_index = anchor_index

            scale_width = math.log(img_width / self.anchors_size[max_iou_index][0])
            scale_height = math.log(img_height / self.anchors_size[max_iou_index][1])
            # 大小物体损失值平衡系数
            scale_adjust_modulu = 2 - scale_width * scale_width

            # 分类标签 label_smooth
            class_index = self.ClassNameToClassIndex[class_name]
            class_list = [self.label_smooth_value / (self.class_num - 1) for i in range(self.class_num)]
            class_list[class_index] = 1 - self.label_smooth_value

            # 定位数据预设
            ground_box = [0, 0, scale_width, scale_height, scale_adjust_modulu, self.input_size * xmin, self.input_size * ymin, self.input_size * xmax, self.input_size * ymax]
            ground_box.extend(class_list)

            if max_iou_index < 3:
                # 已经使用过的需要标记
                small_anchor_mark[small_indexRow][small_indexCol] = 1
                # 定位数据
                center_y = centerY * small_feature_size
                ground_box[1] = offset_y = center_y - small_indexRow
                center_x = centerX * small_feature_size
                ground_box[0] = offset_x = center_x - small_indexCol
                small_ground_truth[small_indexRow][small_indexCol] = np.array(ground_box)

            elif max_iou_index < 6:
                middle_anchor_mark[middle_indexRow][middle_indexCol] = 1
                center_y = centerY * middle_feature_size
                ground_box[1] = offsetY = center_y - middle_indexRow
                center_x = centerX * middle_feature_size
                ground_box[0] = offsetX = center_x - middle_indexCol
                middle_ground_truth[middle_indexRow][middle_indexCol] = np.array(ground_box)

            else:
                big_anchor_mark[big_indexRow][big_indexCol] = 1
                center_y = centerY * big_feature_size
                ground_box[1] = offsetY = center_y - big_indexRow
                center_x = centerX * big_feature_size
                ground_box[0] = offsetX = center_x - big_indexCol
                big_ground_truth[big_indexRow][big_indexCol] = np.array(ground_box, dtype=np.float)

        return small_anchor_mark, small_ground_truth, middle_anchor_mark, middle_ground_truth, big_anchor_mark, big_ground_truth
        '''

'''
train_dataSet = VOCDataSet(imgs_path="../DataSet/VOC2007+2012/Train/JPEGImages",annotations_path="../DataSet/VOC2007+2012/Train/Annotations",classes_file="../DataSet/VOC2007+2012/class.data", is_train=True, class_num=20)
#2011_006556.xml img has block
#2011_005035.xml
base_img_size = 608
anchor_boxes = [[10, 11], [15, 28], [36, 22], [30, 60], [61, 125], [67, 46], [129, 88], [162, 211], [391, 336]]
train_dataSet.setInputSize(base_img_size, anchor_boxes)
train_dataSet.read_img("../DataSet/VOC2007+2012/Train/JPEGImages/2008_001230.jpg", "../DataSet/VOC2007+2012/Train/Annotations/2008_001230.xml")
'''
'''
big assign row:9 col:5
big assign row:10 col:11
big assign row:10 col:16
big assign row:12 col:1
big assign row:13 col:16
big assign row:15 col:9

big row:9 col:5 xmin:92 ymin:179 xmax:283 ymax:455 class_index:14
big row:12 col:1 xmin:1 ymin:339 xmax:111 ymax:447 class_index:8

middle assign row:18 col:1
middle assign row:18 col:5
middle assign row:27 col:1
middle assign row:27 col:9
middle assign row:28 col:13
middle assign row:29 col:19

row:18 col:1 middle xmin:1 ymin:258 xmax:34 ymax:333 class_index:8
row:18 col:5 middle xmin:51 ymin:277 xmax:117 ymax:306 class_index:8
row:27 col:1 middle xmin:1 ymin:377 xmax:47 ymax:503 class_index:4
row:27 col:9 middle xmin:129 ymin:415 xmax:169 ymax:474 class_index:4
row:28 col:13 middle xmin:190 ymin:405 xmax:227 ymax:512 class_index:4

'''



