import numpy as np

def iou(cluster, boxes):
    Area_culster = cluster[0] * cluster[1]
    Area_boxes = boxes[:,0] * boxes[:,1]
    Area_inter = np.minimum(cluster[0], boxes[:,0]) * np.minimum(cluster[1], boxes[:,1])
    return Area_inter / (Area_culster + Area_boxes - Area_inter)

def kmeans(boxes, k, dist=np.median, seed=1):
    """
    计算k-均值聚类与交集的联合(IoU)指标
    :param boxes:形状(r, 2)的numpy数组，其中r是行数
    :param k: 集群的数量
    :param dist: 距离函数
    :返回:形状的numpy数组(k, 2)
    """
    rows = boxes.shape[0] # 样本数

    distances = np.empty((rows, k))  # N row x N cluster  distance[row][k]:第row个样本到第k个聚类中心的距离
    last_clusters = np.zeros((rows,))

    np.random.seed(seed) # 设置随机种子

    # 将集群中心初始化为k个项 np.random.choice(rows, k, replace=False) 从0~rows-1的均匀分布中随机采样k个点并保证不重复
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        # 为每个点指定聚类的类别（如果这个点距离某类别最近，那么就指定它是这个类别)
        for icluster in range(k):  # I made change to lars76's code here to make the code faster
            distances[:, icluster] = 1 - iou(clusters[icluster], boxes)

        nearest_clusters = np.argmin(distances, axis=1) # 找到每一个样本距离最近的聚类中心
        # 如果聚类簇的中心位置基本不变了，那么迭代终止。
        if (last_clusters == nearest_clusters).all(): # 所有的聚类中心不变
            break

        # 重新计算每个聚类簇的平均中心位置，并它作为聚类中心点
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0) #聚类中心对每个维度求均值

        last_clusters = nearest_clusters

    return clusters, nearest_clusters, distances


import os
import cv2
import time
import image
target_size = 608
k = 9
txts_path = "../DataSet/COCO2017/Train/Labels"
imgs_path = "../DataSet/COCO2017/Train/Images"
txts_name = os.listdir(txts_path)
bounding_boxes = []

for txt_name in txts_name:
    img_path = os.path.join(imgs_path, txt_name.replace(".txt", ".jpg"))
    img = cv2.imread(img_path)

    coords = []
    with open(os.path.join(txts_path, txt_name), 'r') as file:
        for line_context in file:
            line_context = line_context.split(' ')

            class_id = int(line_context[4])
            xmin = round(float(line_context[0]))
            ymin = round(float(line_context[1]))
            xmax = round(float(line_context[2]))
            ymax = round(float(line_context[3]))
            coords.append([xmin, ymin, xmax, ymax, class_id])
    img, coords = image.resize_image_with_coords(img, target_size, target_size, coords)

    for coord in coords:
        coord[0] = round(coord[0] * target_size)
        coord[1] = round(coord[1] * target_size)
        coord[2] = round(coord[2] * target_size)
        coord[3] = round(coord[3] * target_size)
        box = [coord[2] - coord[0], coord[3] - coord[1]]
        bounding_boxes.append(box)

clusters, nearest_clusters, distances = kmeans(np.array(bounding_boxes), k, seed=int(time.time()))
import matplotlib.pyplot as plt
colors = ['peru', 'dodgerblue', 'turquoise', 'brown', 'red', 'lightsalmon', 'orange', 'springgreen' , 'orchid']
point_x = [list() for i in range(k)]
point_y = [list() for i in range(k)]

'''
for index in range(len(nearest_clusters)):
    point_x[nearest_clusters[index]].append(bounding_boxes[index][0])
    point_y[nearest_clusters[index]].append(bounding_boxes[index][1])

for cluster_index in range(k):
    plt.scatter(point_x[cluster_index], point_y[cluster_index], color=colors[cluster_index])
'''
'''
for box_index in range(len(bounding_boxes)):
    if box_index > 20000:
        break
    color = colors[nearest_clusters[box_index]]
    plt.scatter(bounding_boxes[box_index][0], bounding_boxes[box_index][1], color=color)
'''
clusters.sort(lambda: x[0] * x[1] for x in clusters)
plt.show()
print(clusters)








