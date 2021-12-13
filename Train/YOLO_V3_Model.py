import torch.nn as nn
import torch.nn.functional as F
import torch
from DarkNet53 import CBL, ResUnit, ResX

class ConventionSet(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2, out_channels_3, out_channels_4, out_channels):
        self.conv = nn.Sequential(
            CBL(in_channels, out_channels_1, 1, 1, 0),
            CBL(out_channels_1, out_channels_2, 3, 1, 1),
            CBL(out_channels_2, out_channels_3, 1, 1, 0),
            CBL(out_channels_3, out_channels_4, 3, 1, 1),
            CBL(out_channels_4, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.conv(x)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, CBL):
                m.weight_init()

class UpSampleLayer(nn.Module):
    def __init__(self):
        super(UpSampleLayer, self).__init__()
    def forward(self,x):
        return F.interpolate(x,scale_factor=2,mode='nearest')

class YOLO_Block(nn.Module):
    def __init__(self, in_channels, out_channels, classes_num, need_branch = True):
        super(YOLO_Block, self).__init__()
        self.need_branch = need_branch
        self.conv = nn.Sequential(
            CBL(in_channels, out_channels, 1, 1, 0),
            CBL(out_channels, in_channels, 3, 1, 1),
            CBL(in_channels, out_channels, 1, 1, 0),
            CBL(out_channels, in_channels, 3, 1, 1),
            CBL(in_channels, out_channels, 1, 1, 0),
        )
        self.predict = nn.Sequential(
            CBL(out_channels, in_channels, 3, 1, 1),
            CBL(in_channels, 3 * (5 + classes_num), 1, 1, 0, inplace=False),
        )


    def forward(self, x):
        if self.need_branch:
            x = self.conv(x)
            x_predict = self.predict(x)
            return x, x_predict
        else:
            x = self.predict(self.conv(x))
            return x

class YOLO_V3(nn.Module):
    def __init__(self,class_num=80):
        super(YOLO_V3,self).__init__()
        self.class_num = class_num

        self.conv_pre = nn.Sequential(
            CBL(3, 32, 3, 1, 1),
            CBL(32, 64, 3, 2, 1),
        )

        self.Res_1_64 = ResX(64, 32, 1, 1, 0, 64, 3, 1, 1)
        self.Res_2_128 = nn.Sequential(
            CBL(64, 128, 3, 2, 1),
            ResX(128, 64, 1, 1, 0, 128, 3, 1, 1),
            ResX(128, 64, 1, 1, 0, 128, 3, 1, 1),
        )
        self.Res_8_256 = nn.Sequential(
            CBL(128, 256, 3, 2, 1),
            ResX(256, 128, 1, 1, 0, 256, 3, 1, 1),
            ResX(256, 128, 1, 1, 0, 256, 3, 1, 1),
            ResX(256, 128, 1, 1, 0, 256, 3, 1, 1),
            ResX(256, 128, 1, 1, 0, 256, 3, 1, 1),
            ResX(256, 128, 1, 1, 0, 256, 3, 1, 1),
            ResX(256, 128, 1, 1, 0, 256, 3, 1, 1),
            ResX(256, 128, 1, 1, 0, 256, 3, 1, 1),
            ResX(256, 128, 1, 1, 0, 256, 3, 1, 1),
        )

        self.Res_8_512 = nn.Sequential(
            CBL(256, 512, 3, 2, 1),
            ResX(512, 256, 1, 1, 0, 512, 3, 1, 1),
            ResX(512, 256, 1, 1, 0, 512, 3, 1, 1),
            ResX(512, 256, 1, 1, 0, 512, 3, 1, 1),
            ResX(512, 256, 1, 1, 0, 512, 3, 1, 1),
            ResX(512, 256, 1, 1, 0, 512, 3, 1, 1),
            ResX(512, 256, 1, 1, 0, 512, 3, 1, 1),
            ResX(512, 256, 1, 1, 0, 512, 3, 1, 1),
            ResX(512, 256, 1, 1, 0, 512, 3, 1, 1),
        )

        self.Res_4_1024 = nn.Sequential(
            CBL(512, 1024, 3, 2, 1),
            ResX(1024, 512, 1, 1, 0, 1024, 3, 1, 1),
            ResX(1024, 512, 1, 1, 0, 1024, 3, 1, 1),
            ResX(1024, 512, 1, 1, 0, 1024, 3, 1, 1),
            ResX(1024, 512, 1, 1, 0, 1024, 3, 1, 1),
        )

        #self.big_yolo = YOLO_Block(1024, 512, class_num)

        self.bigger_detect = nn.Sequential(
            CBL(1024, 512, 1, 1, 0),
            CBL(512, 1024, 3, 1, 1),
            CBL(1024, 512, 1, 1, 0),
            CBL(512, 1024, 3, 1, 1),
            CBL(1024, 512, 1, 1, 0),
            CBL(512, 1024, 3, 1, 1),
            CBL(1024, 3 * (5 + class_num), 1, 1, 0, inplace=False),
        )

        self.neck_bigger_middle = nn.Sequential(
            CBL(512, 256, 1, 1, 0),
            UpSampleLayer(),
        )
        #self.middle_yolo = YOLO_Block(786, 256, class_num)

        self.middle_detect = nn.Sequential(
            CBL(768, 256, 1, 1, 0),
            CBL(256, 512, 3, 1, 1),
            CBL(512, 256, 1, 1, 0),
            CBL(256, 512, 3, 1, 1),
            CBL(512, 256, 1, 1, 0),
            CBL(256, 512, 3, 1, 1),
            CBL(512, 3 * (5 + class_num), 1, 1, 0, inplace=False),
        )

        self.neck_middle_small = nn.Sequential(
            CBL(256, 128, 1, 1, 0),
            UpSampleLayer(),
        )
        #self.small_yolo =

        self.small_detect = nn.Sequential(
            CBL(384, 128, 1, 1, 0),
            CBL(128, 256, 3, 1, 1),
            CBL(256, 128, 1, 1, 0),
            CBL(128, 256, 3, 1, 1),
            CBL(256, 128, 1, 1, 0),
            CBL(128, 256, 3, 1, 1),
            CBL(256, 3 * (5 + self.class_num), 1, 1, 0, inplace=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        def branch(layer_weight, layer_in):
            for i, layer in enumerate(layer_weight):
                layer_in = layer(layer_in)
                if i == 4:
                    out_branch = layer_in
            return layer_in, out_branch

        x = self.conv_pre(x)
        x = self.Res_1_64(x)
        x = self.Res_2_128(x)

        x_small = self.Res_8_256(x)

        x_middle = self.Res_8_512(x_small)

        x_bigger = self.Res_4_1024(x_middle)

        # big target
        x_predict_bigger, x_bigger_branch = branch(self.bigger_detect, x_bigger)
        batch_size, channels, width, height = x_predict_bigger.size()
        x_predict_bigger = x_predict_bigger.permute(0, 2, 3, 1)
        x_predict_bigger = x_predict_bigger.view([batch_size, width, height, 3, 5 + self.class_num])
        #x_predict_bigger = torch.cat([self.sigmoid(x_predict_bigger[...,0:2]), x_predict_bigger[...,2:4], self.sigmoid(x_predict_bigger[...,4:])], dim=4)
        x_predict_bigger[...,0:2] = self.sigmoid(x_predict_bigger[...,0:2])
        x_predict_bigger[...,4] = self.sigmoid(x_predict_bigger[...,4])

        # middle target
        x_bigger_branch = self.neck_bigger_middle(x_bigger_branch)
        x_middle = torch.cat(tensors=[x_middle, x_bigger_branch], dim=1)
        x_predict_middle, x_middle_branch = branch(self.middle_detect, x_middle)
        batch_size, channels, width, height = x_predict_middle.size()
        x_predict_middle = x_predict_middle.permute(0, 2, 3, 1)
        x_predict_middle = x_predict_middle.view([batch_size, width, height, 3, 5 + self.class_num])
        x_predict_middle[..., 0:2] = self.sigmoid(x_predict_middle[..., 0:2])
        x_predict_middle[..., 4] = self.sigmoid(x_predict_middle[..., 4])

        # small target
        x_middle_branch = self.neck_middle_small(x_middle_branch)
        x_small = torch.cat(tensors=[x_small, x_middle_branch], dim=1)
        x_predict_small = self.small_detect(x_small)
        batch_size, channels, width, height = x_predict_small.size()
        x_predict_small = x_predict_small.permute(0, 2, 3, 1)
        x_predict_small = x_predict_small.view([batch_size, width, height, 3, 5 + self.class_num])
        x_predict_small[..., 0:2] = self.sigmoid(x_predict_small[..., 0:2])
        x_predict_small[..., 4] = self.sigmoid(x_predict_small[..., 4])

        return x_predict_small, x_predict_middle, x_predict_bigger

    # 定义权值初始化
    def initialize_weights(self, pre_weight_file, isFreezed = True):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, CBL):
                m.weight_init()

        net_param_dict = torch.load(pre_weight_file, map_location=torch.device("cpu"))['min_loss_model']
        self_param_dict = self.state_dict()
        for name, layer in self.named_parameters():
            if name in net_param_dict:
                self_param_dict[name] = net_param_dict[name]
                #layer.weight = net_param_dict[name]
                #layer.requires_grad = not isFreezed
                #print("name:{} layer:{} dict-content:{}".format(name, layer, net_param_dict[name]))
        self.load_state_dict(self_param_dict)
