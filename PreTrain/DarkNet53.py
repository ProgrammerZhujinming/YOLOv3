import torch
import torch.nn as nn

class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride, padding, inplace=True):
        super(CBL, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernal_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=inplace),
        )

    def forward(self, x):
        return self.conv(x)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size = 3, stride = 1, padding = 1):
        super(ResUnit,self).__init__()
        self.conv_feature = nn.Sequential(
            CBL(in_channels, out_channels, kernal_size, stride, padding),
            CBL(out_channels, out_channels, kernal_size, stride, padding)
        )
        self.conv_redisual = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x_redisual = self.conv_redisual(x)
        x = self.conv_feature(x)
        x = torch.add(x, x_redisual)
        return x

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, CBL):
                m.weight_init()
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)


class ResX(nn.Module):
    def __init__(self, in_channels, out_channels_1, kernal_size_1, stride_1, padding_1, out_channels_2, kernal_size_2, stride_2, padding_2):
        super(ResX,self).__init__()
        self.conv = nn.Sequential(
            CBL(in_channels, out_channels_1, kernal_size_1, stride_1, padding_1),# down sample
            ResUnit(out_channels_1, out_channels_2, kernal_size_2, stride_2, padding_2),
        )

    def forward(self, x):
        return self.conv(x)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, CBL):
                m.weight_init()
            elif isinstance(m, ResUnit):
                m.weight_init()

class DarkNet53(nn.Module):
    def __init__(self, class_num):
        super(DarkNet53, self).__init__()

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

        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.predict = nn.Linear(1024, class_num)

    def forward(self, x):
        x = self.conv_pre(x)
        x = self.Res_1_64(x)
        x = self.Res_2_128(x)
        x = self.Res_8_256(x)
        x = self.Res_8_512(x)
        x = self.Res_4_1024(x)
        x = self.global_pooling(x)
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = self.predict(x)
        return x

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, ResX):
                m.weight_init()
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)