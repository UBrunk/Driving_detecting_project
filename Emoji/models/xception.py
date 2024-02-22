"""
PyTorch 模型文档

该模块提供了用于图像分类任务的 PyTorch 模型。

可用模型：
- mini_xception: XCEPTION 模型的较小版本。

"""

from torch import nn
import torch.nn.functional as F

class mini_xception(nn.Module):
    """
       mini_xception 类是一个 PyTorch 模型类，用于实现 XCEPTION 模型的较小版本。

       参数：
           无

       属性：
           num_channels (int): 输入图像的通道数，默认为 1。
           image_size (int): 输入图像的尺寸，默认为 48。
           num_labels (int): 分类任务的类别数，默认为 7。

       方法：
           __init__: 类的初始化方法，定义模型的结构。
           forward: 前向传播方法，用于定义模型的前向计算过程。
    """
    def __init__(self):
        """
            初始化 mini_xception 类。

            参数：
                无

            返回：
                无
        """
        super(mini_xception,self).__init__()
        self.num_channels=1
        self.image_size=48
        self.num_labels=7
        self.conv2d_1 =nn.Conv2d(in_channels=46,out_channels=8,kernel_size=3,stride=1)
        self.batch_normalization_1=nn.BatchNorm1d(46)
        self.conv2d_2=nn.Conv2d(46,8,3,1)
        self.batch_normalization_2=nn.BatchNorm1d(46)

        #module 1




    def forward(self, x):
        x=F.relu(self.batch_normalization_1)
        x=F.relu(self.batch_normalization_2)
        return x

if __name__ == '__main__':
    print(mini_xception())