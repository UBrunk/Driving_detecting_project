# 导入依赖库
import os.path as osp

# 定义先验框的尺寸列表，对应不同特征图级别
sk = [15, 30, 60, 111, 162, 213, 264]

# 定义不同网络阶段的特征图大小列表
feature_map = [38, 19, 10, 5, 3, 1]

# 定义每个特征图级别的步长（像素）
steps = [8, 16, 32, 64, 100, 300]

# 输入图像的尺寸
image_size = 300

# 默认框的不同长宽比列表，每个列表对应一个特征图级别
aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

# 图像归一化所使用的均值
MEANS = (104, 117, 123)

# 训练时的批处理大小
batch_size = 4

# 数据加载的线程数
data_load_number_worker = 0

# 训练的学习率
lr = 5e-4

# 优化器的动量参数
momentum = 0.9

# 正则化的权重衰减参数
weight_decacy = 5e-4

# 学习率调整的gamma参数
gamma = 0.1

# VOC数据集的根目录
VOC_ROOT = osp.join('./', "dataset/")

# 数据集的根目录别名
dataset_root = VOC_ROOT

# 是否使用CUDA进行训练
use_cuda = True

# 学习率调整的步数
lr_steps = (8000, 10000, 12000)

# 训练的最大迭代次数
max_iter = 120000

# 数据集中的类别数目
class_num = 7

# 另一个表示类别数目的整数
class_num2 = 7


# import os.path as osp
# sk = [ 15, 30, 60, 111, 162, 213, 264 ]
# feature_map = [ 38, 19, 10, 5, 3, 1 ]
# steps = [ 8, 16, 32, 64, 100, 300 ]
# image_size = 300
# aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
# MEANS = (104, 117, 123)
# batch_size = 4
# data_load_number_worker = 0
# lr = 5e-4
# momentum = 0.9
# weight_decacy = 5e-4
# gamma = 0.1
# VOC_ROOT = osp.join('./', "dataset/")
# dataset_root = VOC_ROOT
# use_cuda = True
# # lr_steps = (80000, 100000, 120000)
# lr_steps = (8000, 10000, 12000)
# max_iter = 120000
# class_num = 7
# class_num2= 7