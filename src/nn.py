from torch import nn
import torch
from utils import make_anchors


class Conv(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        k=3,
        s=1,
        p=1,
        g=1,
        act=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            c1,
            c2,
            k,
            s,
            p,
            bias=False,
            groups=g,
        )
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# 2.1 Bottleneck: stack of 2 Conv with shortcut connection (True/False)
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.cv1 = Conv(in_channels, out_channels, k=3, s=1, p=1)
        self.cv2 = Conv(out_channels, out_channels, k=3, s=1, p=1)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))


# 2.2 C2f: Conv + bottleneck*N+ Conv
class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, n, shortcut=True):
        super().__init__()

        self.c = out_channels // 2  # hidden channels
        self.num_bottlenecks = n

        self.cv1 = Conv(in_channels, 2 * self.c, k=1, s=1, p=0)

        # sequence of bottleneck layers
        self.m = nn.ModuleList([Bottleneck(self.c, self.c, shortcut) for _ in range(n)])

        self.cv2 = Conv(
            (n + 2) * self.c,
            out_channels,
            k=1,
            s=1,
            p=0,
        )

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        # kernel_size= size of maxpool
        super().__init__()
        hidden_channels = c1 // 2
        self.cv1 = Conv(c1, hidden_channels, k=1, s=1, p=0)
        # concatenate outputs of maxpool and feed to cv2
        self.cv2 = Conv(4 * hidden_channels, c2, k=1, s=1, p=0)

        # maxpool is applied at 3 different scales
        self.m = nn.MaxPool2d(
            kernel_size=k,
            stride=1,
            padding=k // 2,
            dilation=1,
            ceil_mode=False,
        )

    def forward(self, x):
        x = self.cv1(x)

        # apply maxpooling at different scales
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)

        # concatenate
        y = torch.cat([x, y1, y2, y3], dim=1)

        # final conv
        y = self.cv2(y)

        return y


def scales(version):
    if version == "n":
        return 1 / 3, 1 / 4, 2.0
    elif version == "s":
        return 1 / 3, 1 / 2, 2.0
    elif version == "m":
        return 2 / 3, 3 / 4, 1.5
    elif version == "l":
        return 1.0, 1.0, 1.0
    elif version == "x":
        return 1.0, 1.25, 1.0


class Backbone(nn.Module):
    def __init__(self, version, in_channels=3, shortcut=True):
        super().__init__()
        d, w, r = scales(version)

        # conv layers
        self.conv_0 = Conv(in_channels, int(64 * w), k=3, s=2, p=1)
        self.conv_1 = Conv(int(64 * w), int(128 * w), k=3, s=2, p=1)
        self.conv_3 = Conv(int(128 * w), int(256 * w), k=3, s=2, p=1)
        self.conv_5 = Conv(int(256 * w), int(512 * w), k=3, s=2, p=1)
        self.conv_7 = Conv(int(512 * w), int(512 * w * r), k=3, s=2, p=1)

        # c2f layers
        self.c2f_2 = C2f(int(128 * w), int(128 * w), n=int(3 * d), shortcut=True)
        self.c2f_4 = C2f(int(256 * w), int(256 * w), n=int(6 * d), shortcut=True)
        self.c2f_6 = C2f(int(512 * w), int(512 * w), n=int(6 * d), shortcut=True)
        self.c2f_8 = C2f(
            int(512 * w * r),
            int(512 * w * r),
            n=int(3 * d),
            shortcut=True,
        )

        # sppf
        self.sppf = SPPF(int(512 * w * r), int(512 * w * r))

    def forward(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.c2f_2(x)
        x = self.conv_3(x)
        out1 = self.c2f_4(x)  # keep for output
        x = self.conv_5(out1)
        out2 = self.c2f_6(x)  # keep for output
        x = self.conv_7(out2)
        x = self.c2f_8(x)
        out3 = self.sppf(x)  # keep for output
        return out1, out2, out3


class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode="nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode=self.mode
        )


class Neck(nn.Module):
    def __init__(self, version):
        super().__init__()
        d, w, r = scales(version)

        self.up = Upsample()  # no trainable parameters
        self.c2f_1 = C2f(
            in_channels=int(512 * w * (1 + r)),
            out_channels=int(512 * w),
            n=int(3 * d),
            shortcut=False,
        )
        self.c2f_2 = C2f(
            in_channels=int(768 * w),
            out_channels=int(256 * w),
            n=int(3 * d),
            shortcut=False,
        )
        self.c2f_3 = C2f(
            in_channels=int(768 * w),
            out_channels=int(512 * w),
            n=int(3 * d),
            shortcut=False,
        )
        self.c2f_4 = C2f(
            in_channels=int(512 * w * (1 + r)),
            out_channels=int(512 * w * r),
            n=int(3 * d),
            shortcut=False,
        )

        self.cv_1 = Conv(
            c1=int(256 * w),
            c2=int(256 * w),
            k=3,
            s=2,
            p=1,
        )
        self.cv_2 = Conv(
            c1=int(512 * w),
            c2=int(512 * w),
            k=3,
            s=2,
            p=1,
        )

    def forward(self, x_res_1, x_res_2, x):
        # x_res_1,x_res_2,x = output of backbone
        res_1 = x  # for residual connection

        x = self.up(x)
        x = torch.cat([x, x_res_2], dim=1)

        res_2 = self.c2f_1(x)  # for residual connection

        x = self.up(res_2)
        x = torch.cat([x, x_res_1], dim=1)

        out_1 = self.c2f_2(x)

        x = self.cv_1(out_1)

        x = torch.cat([x, res_2], dim=1)
        out_2 = self.c2f_3(x)

        x = self.cv_2(out_2)

        x = torch.cat([x, res_1], dim=1)
        out_3 = self.c2f_4(x)

        return out_1, out_2, out_3


# DFL
class DFL(nn.Module):
    def __init__(self, ch=16):
        super().__init__()

        self.ch = ch

        self.conv = nn.Conv2d(
            in_channels=ch, out_channels=1, kernel_size=1, bias=False
        ).requires_grad_(False)

        # initialize conv with [0,...,ch-1]
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)  # DFL only has ch parameters

    def forward(self, x):
        # x must have num_channels = 4*ch: x=[bs,4*ch,c]
        b, c, a = x.shape  # c=4*ch
        x = x.view(b, 4, self.ch, a).transpose(1, 2)  # [bs,ch,4,a]

        # take softmax on channel dimension to get distribution probabilities
        x = x.softmax(1)  # [b,ch,4,a]
        x = self.conv(x)  # [b,1,4,a]
        return x.view(b, 4, a)  # [b,4,a]


class Head(nn.Module):
    def __init__(self, version, ch=16, num_classes=80):

        super().__init__()
        self.ch = ch  # dfl channels
        self.coordinates = self.ch * 4  # number of bounding box coordinates
        self.nc = num_classes  # 80 for COCO
        self.no = self.coordinates + self.nc  # number of outputs per anchor box

        self.stride = torch.tensor([8.0, 16.0, 32.0])  # default strides for YOLOv8

        d, w, r = scales(version=version)

        # for bounding boxes
        self.box = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(
                        int(256 * w),
                        self.coordinates,
                        k=3,
                        s=1,
                        p=1,
                    ),
                    Conv(
                        self.coordinates,
                        self.coordinates,
                        k=3,
                        s=1,
                        p=1,
                    ),
                    nn.Conv2d(
                        self.coordinates, self.coordinates, kernel_size=1, stride=1
                    ),
                ),
                nn.Sequential(
                    Conv(
                        int(512 * w),
                        self.coordinates,
                        k=3,
                        s=1,
                        p=1,
                    ),
                    Conv(
                        self.coordinates,
                        self.coordinates,
                        k=3,
                        s=1,
                        p=1,
                    ),
                    nn.Conv2d(
                        self.coordinates, self.coordinates, kernel_size=1, stride=1
                    ),
                ),
                nn.Sequential(
                    Conv(
                        int(512 * w * r),
                        self.coordinates,
                        k=3,
                        s=1,
                        p=1,
                    ),
                    Conv(
                        self.coordinates,
                        self.coordinates,
                        k=3,
                        s=1,
                        p=1,
                    ),
                    nn.Conv2d(
                        self.coordinates, self.coordinates, kernel_size=1, stride=1
                    ),
                ),
            ]
        )

        # for classification
        self.cls = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(int(256 * w), self.nc, k=3, s=1, p=1),
                    Conv(self.nc, self.nc, k=3, s=1, p=1),
                    nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1),
                ),
                nn.Sequential(
                    Conv(int(512 * w), self.nc, k=3, s=1, p=1),
                    Conv(self.nc, self.nc, k=3, s=1, p=1),
                    nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1),
                ),
                nn.Sequential(
                    Conv(int(512 * w * r), self.nc, k=3, s=1, p=1),
                    Conv(self.nc, self.nc, k=3, s=1, p=1),
                    nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1),
                ),
            ]
        )

        # dfl
        self.dfl = DFL(ch=self.ch)

    def forward(self, x):
        for i in range(len(self.box)):  # detection head i
            box = self.box[i](x[i])  # [bs,num_coordinates,w,h]
            cls = self.cls[i](x[i])  # [bs,num_classes,w,h]
            x[i] = torch.cat((box, cls), dim=1)  # [bs,num_coordinates+num_classes,w,h]

        # in training, no dfl output
        if self.training:
            return x  # [3,bs,num_coordinates+num_classes,w,h]

        # in inference time, dfl produces refined bounding box coordinates
        self.stride = self.stride.to(x[0].device)  # ensure stride is on the same device
        anchors, strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))

        # concatenate predictions from all detection layers
        x = torch.cat(
            [i.view(x[0].shape[0], self.no, -1) for i in x], dim=2
        )  # [bs, 4*self.ch + self.nc, sum_i(h[i]w[i])]

        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

        a, b = self.dfl(box).chunk(2, 1)  # a=b=[bs,2×self.ch,sum_i(h[i]w[i])]
        a = anchors.unsqueeze(0) - a
        b = anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

        return torch.cat(tensors=(box * strides, cls.sigmoid()), dim=1)


class Yolov8n(nn.Module):
    def __init__(self, version):
        super().__init__()
        self.backbone = Backbone(version=version)
        self.neck = Neck(version=version)
        self.head = Head(version=version)

    def forward(self, x):
        x = self.backbone(x)  # return out1,out2,out3
        x = self.neck(x[0], x[1], x[2])  # return out_1, out_2,out_3
        return self.head(list(x))
