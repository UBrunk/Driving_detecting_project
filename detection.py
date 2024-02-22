import torch
from torch.autograd import Function
from utils import decode, nms


class Detect(Function):
    """
        在测试阶段，Detect是SSD模型的最终层。
        对位置预测进行解码，根据置信度分数和阈值对位置预测进行非最大抑制，
        并选取前top_k个置信度分数和位置的输出预测。

        参数:
            num_classes (int): 类别数目，不包括背景类别
            bkg_label (int): 背景类别的标签索引
            top_k (int): 每个图像中保留的最高预测数量
            conf_thresh (float): 置信度阈值，低于此阈值的预测将被忽略
            nms_thresh (float): 非最大抑制的阈值，用于过滤重叠的边界框

        注意:
            如果nms_thresh小于或等于0，将引发ValueError异常。
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        """
            初始化Detect函数

            参数:
                num_classes (int): 类别数目，不包括背景类别
                bkg_label (int): 背景类别的标签索引
                top_k (int): 每个图像中保留的最高预测数量
                conf_thresh (float): 置信度阈值，低于此阈值的预测将被忽略
                nms_thresh (float): 非最大抑制的阈值，用于过滤重叠的边界框
        """
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = (0.1,0.2)

    def forward(self, loc_data, conf_data, prior_data):
        """
            Detect前向传播函数

            参数:
                loc_data (tensor): 来自位置预测层的位置预测
                    形状: [batch,num_priors*4]
                conf_data (tensor): 来自置信度预测层的置信度预测
                    形状: [batch*num_priors,num_classes]
                prior_data (tensor): 来自先验框层的先验框和方差
                    形状: [1,num_priors,4]

            返回:
                output (tensor): 输出预测
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)

                if count==0:
                    continue
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output
