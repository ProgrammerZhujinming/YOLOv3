# YOLOv3
复现YOLOv3算法。

# Introduction 
个人博客YOLOv3算法详解地址：https://blog.csdn.net/qq_39304630/article/details/121373526?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164870519816780264020848%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=164870519816780264020848&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2  
个人博客YOLOv3论文精读地址：https://blog.csdn.net/qq_39304630/article/details/121341732?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164870519816780264020848%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=164870519816780264020848&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2  

# Project Instruction  
[注]：不在下面列表中写出的部分 是暂时没有特殊用处的部分
--DataSet 数据集  
----COCO2017 COCO2017数据集文件夹  
------Train 训练集  
--------Images 图片  
--------Labels 脚本生成的txt标注文件  
------Val 验证集  
--------Images 图片  
--------Labels 脚本生成的txt标注文件  
----create_cocolabel_from_json.py 用于生成本项目需要的COCO2017预训练数据集的脚本  
--Detection 使用训练出来的模型进行检测  
----YOLOv3_CaptureDetection.py 使用训练出来的模型进行摄像头检测  
----YOLOv3_ImageDetection.py 使用训练出来的模型进行图片检测  
----YOLOv3_VideoDetection.py 使用训练出来的模型进行视频检测  
--PreTrain 预训练  
----log 预训练日志  
----weight 预训练权重文件  
----COCO_Classify.py COCO数据集类,用于分类  
----DarkNet53.py DarkNet53网络定义  
----VOC_Classify.py VOC数据集类  
----Extract_Final_Model.py 获得训练产生的权重文件pth中最优的参数  
--Test 模型最终性能指标测试  
--Train  
----weights 检测训练的权重文件  
----Anchor_Kean_Means.py 通过k-means聚类获得Anchor box的尺寸  
----COCO_DataSet.py COCO数据集用于训练目标检测  
----VOC_DataSet.py VOC数据集用于训练目标检测  
----YOLOv3_Model.py YOLOv3的模型定义  
----YOLOv3_LossFunction.py YOLOv3的损失函数定义  
----Extract_Final_Model.py 提取最终性能最好的模型  
--utils 自定义工具包  
----image 用于处理图像增强的一些方法  
----model 对模型冻结和解冻的一些方法定义  
--DarkNet53-Train.py 预训练  
--DarkNet53-FromRecord.py 预训练中断恢复  
--YOLOv3_Train.py 检测训练  
--YOLOv3_FromRecord.py 检测训练中断恢复   

# Usage  
To install requirements:  pip install -r requirements.txt  
本项目使用的预训练数据集为COCO2017,检测训练使用的是VOC数据集。预训练入口为DarkNet53-Train.py,检测训练入口为YOLO_V3_Train.py。  

// YOLOv3DarkNet53预训练

// 下载项目  
git clone git@github.com:ProgrammerZhujinming/YOLOv1.git  

//anaconda虚拟环境创建
conda env create -f DeepLearning.yaml

// COCO2017数据集下载
wget -c http://images.cocodataset.org/zips/train2017.zip  
unzip train2017.zip > /dev/null  
rm -f train2017.zip  

wget -c http://images.cocodataset.org/zips/val2017.zip  
unzip val2017.zip > /dev/null  
rm -f val2017.zip  

wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip  
unzip annotations_trainval2017.zip >> /dev/null  
rm -f annotations_trainval2017.zip  

// 使用脚本生成训练用的COCO2017的数据集  
python ./YOLOv3/DataSet/create_cocolabel_from_json.py --json_file="./annotations/instances_train2017.json" --class_file="./YOLOv1/DataSet/COCO2017/class.txt" --imgs_path="./train2017" --target_labels_path="./YOLOv1/DataSet/COCO2017/Train/Labels"  
python ./YOLOv3/DataSet/create_cocolabel_from_json.py --json_file="./annotations/instances_val2017.json" --class_file="./YOLOv1/DataSet/COCO2017/class.txt" --imgs_path="./val2017" --target_labels_path="./YOLOv1/DataSet/COCO2017/Val/Labels"  

// 清理下载的COCO2017数据  
rm -r annotations  
rm -r train2017  
rm -r val2017  

// YOLOv3目标检测训练  
// VOC目标检测数据集下载  
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar  
tar -x VOCtrainval_11-May-2012.tar  
rm -f VOCtrainval_11-May-2012.tar  

wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar  
tar -x VOCtrainval_06-Nov-2007.tar  
rm -f VOCtrainval_06-Nov-2007.tar  

//生成目标检测数据集
python split_voc_train_val.py --voc_pth="../../VOCdevkit/VOC2007" --train_ratio="0.9"  --target_train_pth="./YOLOv3/DataSet/VOC2007+2012/Train" --target_val_pth="./YOLOv3/DataSet/VOC2007+2012/Val"
python split_voc_train_val.py --voc_pth="../../VOCdevkit/VOC2012" --train_ratio="0.9" --target_train_pth="./YOLOv3/DataSet/VOC2007+2012/Train" --target_val_pth="./YOLOv3/DataSet/VOC2007+2012/Val"

//DarkNet预训练
cd ./YOLOv3
python DarkNet53-Train.py

//YOLOv3训练  注意您需要从保存的权重文件中提取最优的预训练部分 
cd ./PreTrain
python Extract_Final_Model.py
cd ..
python YOLOv3_Train.py

# Evaluation  
Loss/Accuracy等训练指标使用tensorboardX保存在日志文件里,性能存在波动是因为YOLOv3的多尺度训练技巧导致的，但是即便如此，性能的提升趋势还是显而易见的。

# Pre-trained Models  
预训练模型保存在PreTrain/weights里。  

# Results
模型结果  
![avatar](./results/1.png)  

![avatar](./results/2.png)  

![avatar](./results/3.png)  

![avatar](./results/4.png)  

# Contributing







# YOLOv3 Update Record
项目说明：  
1.tensorboard功能需要PyTorch版本在1.1.0及以上，在项目目录下执行指令tensorboard --logdir=log即可启动(如果出现无法找到命令的错误，则可能需要安装tensorflow)  
2.相应需要查看自己的cuda版本是否支持对应的PyTorch版本  
3.目前数据增强方案为在训练中采用随机种子进行增强，并且是一张图片一次随机种子。
4.本项目为个人复现论文研究使用，禁止任何商业用途，如需转载，请附带地址：https://github.com/ProgrammerZhujinming/YOLO_V3.git  谢谢配合  
5.项目代码部分说明：https://blog.csdn.net/qq_39304630/article/details/121373526

使用说明:  
1.本项目注重的是对YOLO v3原文的复现，训练的入口为Train\YOLO_V3_Train.py  
2.YOLO_V3_FromRecord.py用于从中断的训练中恢复  

## 更新日志 11-2   
1.本项目没有完全按照YOLO v3论文的说明来，尤其是对于负样本向anchor靠齐的情况，目前尚不能理解。  
2.特征图输出功能--该功能严重影响训练速度，如有需要请自行开启：# feature_map_visualize(batch_train[0], writer)，取消注释即可。当然如果想要看到特征图又怕影响训练速度，可以选择将特征图的显示放在每一个epoch而不是batch中。  
3.使用特征图功能可能会遇到TypeError: clamp_(): argument ‘min’ must be Number, not Tensor，此时需要修改torchvison.utils源码，将norm_ip(t, t.min(), t.max())改为norm_ip(t, float(t.min()))  
4.mAP脚本笔者暂时还没写好。  
5.笔者使用的数据集太大了会爆仓，会传到网盘上进行分享，上传完毕会在csdn博客上贴出，如果过期了可通过csdn联系我。当然如果您愿意，可以自行下载数据集，笔者也提供了可以从coco官方的json标注文件中提取bbox信息的脚本DataSet/create_cocolabel_from_json.py。  
6.笔者使用COCO2017数据集进行预训练，使用VOC07+12进行目标检测的训练。  

## 更新日志 12-13   
1.笔者使用的ingore_threshold为0.3。  
2.在计算loss时，笔者对正负样本做了样本数量上的平均而不是batch_size的平均，避免正负样本的极端不平衡，对于负样本，没有采用作者所说的先向anchor学习的方式，同时引入self.noobj=3，是为了平衡正样本拥有定位、置信度、分类三部分损失和负样本只有置信度损失带来的差异。  
3.在各项指标输出时，小、中、大个尺度的指标分别输出。  
4.笔者已经训练了backbone部分，YOLOv3部分已经训练了一些，当前还存在负样本训练不足的情况。  
5.多尺度训练方式不再是每隔10个epoch切换而是每隔10个batch切换。  
6.当前Detection中的视频、摄像头、图像检测已经全部适配笔者实现的YOLOv3。  
7.修复了平均iou显示错误的问题。  

## 更新日志 3-3
1.笔者突然发现自己实现的YOLOv3存在一个问题，就是在predict部分使用Conv2d实际上不应该使用BN层来拉回分布，即在三种尺度下两个用于预测的卷积层是不带bn的，这可能会影响预测结果，已做调整。
