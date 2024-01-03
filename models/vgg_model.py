import torch
import torch.nn as nn

# class VGGBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, num_convs):
#         super(VGGBlock, self).__init__()
#         layers = []
#         for _ in range(num_convs):
#             layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
#             layers.append(nn.ReLU(inplace=True))
#             in_channels = out_channels
#         layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
#         self.vgg_block = nn.Sequential(*layers)
    
#     def forward(self, x):
#         return self.vgg_block(x)

# VGG模型
class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=3, padding=1),  # 输入通道数为1，输出通道数为64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化层

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),  # 输入特征大小需要根据上一层的输出调整
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 10)  # 最后输出10个类别（MNIST数据集有10个类别）
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 创建VGG模型实例
vgg_model = VGG()
