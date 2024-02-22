import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.init as init
import Config
class L2Norm(nn.Module):
    """
        L2范数归一化层，用于对输入进行L2范数归一化处理。

        参数:
            n_channels (int): 输入张量的通道数
            scale (float): 缩放因子，用于初始化权重，如果为None，则不进行缩放
    """
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        if Config.use_cuda:
            self.weight = nn.Parameter(torch.Tensor(self.n_channels).cuda())
        else:
            self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        """
            初始化权重参数为指定的缩放因子。
        """
        nn.init.constant_(self.weight,self.gamma)

    def forward(self, x):
        """
            前向传播函数

            参数:
                x (tensor): 输入张量
            返回:
                out (tensor): 归一化后的张量
        """
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out
