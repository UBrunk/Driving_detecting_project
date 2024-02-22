"""
VOC 数据集类

原作者：Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

更新作者：Ellis Brown, Max deGroot
"""
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
#
VOC_CLASSES = [  # always index 0
		'open_eye','closed_eye','closed_mouth','open_mouth','calling','smoke']
# VOC_CLASSES2 = [  # always index 0
# 		'open_eye','closed_eye','closed_mouth','open_mouth','calling','smoke']
# note: if you used our download scripts, this should be right


class VOCAnnotationTransform(object):
    """
        将 VOC 标注转换为边界框坐标和标签索引的张量
       使用类名到索引的字典进行初始化

       参数:
           class_to_ind (dict, optional): 类名到索引的字典
               (默认: VOC 20 个类的字母索引)
           keep_difficult (bool, optional): 是否保留困难样本
               (默认: False)
           height (int): 图像高度
           width (int): 图像宽度
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
            参数:
                target (annotation) : 待处理的目标标注，类型为 ET.Element
            返回:
                包含边界框坐标和类别索引的列表  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """
        VOC 检测数据集类

        输入为图像，目标为标注

        参数:
            root (string): VOCdevkit 文件夹的路径.
            image_set (string): 使用的图像集 (例如 'train', 'val', 'test')
            transform (callable, optional): 对输入图像的转换
            target_transform (callable, optional): 对目标标注的转换
            dataset_name (string, optional): 加载的数据集名称
    """

    def __init__(self, root,
                 image_sets=[('trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='My_Data'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (name) in image_sets:
            #rootpath = osp.join(self.root, 'VOC' + year)
            rootpath=self.root
            rootpath='dataset/dataset/'
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        """
        以 PIL 形式返回索引处的原始图像对象

            注意: 不使用 self.__getitem__()，因为传入的任何转换都可能破坏此功能。

            参数:
                index (int): 要显示的图像的索引
            返回:
                PIL 形式的图像
        """

        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        """
            返回索引处图像的原始标注

            注意: 不使用 self.__getitem__()，因为传入的任何转换都可能破坏此功能。

            参数:
                index (int): 要获取标注的图像的索引
            返回:
                列表:  [img_id, [(label, bbox coords),...]]
                    例如: ('001718', [('dog', (96, 13, 438, 332))])
        """
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        """
            以张量形式返回索引处的原始图像

            注意: 不使用 self.__getitem__()，因为传入的任何转换都可能破坏此功能。

            参数:
                index (int): 要显示的图像的索引
            返回:
                张量化的图像版本，已压缩
        """
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
