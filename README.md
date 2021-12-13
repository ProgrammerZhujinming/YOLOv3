# YOLO_V3
环境要求：  
1.PyTorch >= 1.1.0  
2.tensorboardX包  
3.cuda >= 9.2(请注意cuda与pytorch的适配)  
项目说明：  
1.tensorboard功能需要PyTorch版本在1.1.0及以上，在项目目录下执行指令tensorboard --logdir=log即可启动(如果出现无法找到命令的错误，则可能需要安装tensorflow)  
2.相应需要查看自己的cuda版本是否支持对应的PyTorch版本  
3.目前数据增强方案为在训练中采用随机种子进行增强，并且是一张图片一次随机种子。
4.本项目为个人复现论文研究使用，禁止任何商业用途，如需转载，请附带地址：https://github.com/ProgrammerZhujinming/YOLO_V3.git  谢谢配合  
5.项目代码部分说明：https://blog.csdn.net/qq_39304630/article/details/121373526

使用说明:  
1.本项目注重的是对YOLO v3原文的复现，训练的入口为Train\YOLO_V3_Train.py  
2.YOLO_V3_FromRecord.py用于从中断的训练中恢复  

# 更新日志 11-2   
1.本项目没有完全按照YOLO v3论文的说明来，尤其是对于负样本向anchor靠齐的情况，目前尚不能理解。  
2.特征图输出功能--该功能严重影响训练速度，如有需要请自行开启：# feature_map_visualize(batch_train[0], writer)，取消注释即可。当然如果想要看到特征图又怕影响训练速度，可以选择将特征图的显示放在每一个epoch而不是batch中。  
3.使用特征图功能可能会遇到TypeError: clamp_(): argument ‘min’ must be Number, not Tensor，此时需要修改torchvison.utils源码，将norm_ip(t, t.min(), t.max())改为norm_ip(t, float(t.min()))  
4.mAP脚本笔者暂时还没写好。  
5.笔者使用的数据集太大了会爆仓，会传到网盘上进行分享，上传完毕会在csdn博客上贴出，如果过期了可通过csdn联系我。当然如果您愿意，可以自行下载数据集，笔者也提供了可以从coco官方的json标注文件中提取bbox信息的脚本DataSet/create_cocolabel_from_json.py。  
6.笔者使用COCO2017数据集进行预训练，使用VOC07+12进行目标检测的训练。  

# 更新日志 12-13   
1.笔者使用的ingore_threshold为0.3。  
2.在计算loss时，笔者对正负样本做了样本数量上的平均而不是batch_size的平均，避免正负样本的极端不平衡，对于负样本，没有采用作者所说的先向anchor学习的方式，同时引入self.noobj=3，是为了平衡正样本拥有定位、置信度、分类三部分损失和负样本只有置信度损失带来的差异。  
3.在各项指标输出时，小、中、大个尺度的指标分别输出。  
4.笔者已经训练了backbone部分，YOLOv3部分已经训练了一些，当前还存在负样本训练不足的情况。  
5.多尺度训练方式不再是每隔10个epoch切换而是每隔10个batch切换。  
6.当前Detection中的视频、摄像头、图像检测已经全部适配笔者实现的YOLOv3。  
7.修复了平均iou显示错误的问题。  