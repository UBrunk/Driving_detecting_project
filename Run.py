import datetime

from PyQt5 import QtWidgets
from qimage2ndarray.dynqt import QtGui

from mtcnn.detector import detect_faces, show_bboxes, get_face_expression, get_head_pose, get_emotion, get_face_state
from MainWindow import Ui_MainWindow
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtCore import QTimer, QCoreApplication, QDateTime
from PyQt5.QtGui import QPixmap, QImage, QTextCursor
import qimage2ndarray
from torch.autograd import *
from detection import *
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from ssd_net_vgg import *
from voc0712 import *
import torch
import torch.nn as nn
import numpy as np
import cv2
import utils
import torch.backends.cudnn as cudnn
import time


class CameraShow(QMainWindow, Ui_MainWindow):
    def __del__(self):
        """
            析构函数，用于释放摄像头资源。

            在对象被销毁时自动调用。尝试释放摄像头资源，如果释放过程中出现异常，则不执行任何其他操作。
        """
        try:
            self.camera.release()  # 释放资源
        except:
            return

    def __init__(self, parent=None):
        """
            初始化函数，用于初始化相机显示界面。

            参数:
                parent: 父对象，默认为None。

            初始化界面及相关参数，包括设置定时器，连接定时器的超时信号到相应的槽函数，准备相机，初始化Torch模型，
            设置回调函数，显示当前时间，初始化case为0，设置frag_cap为True，设置视频标志为True，初始化颜色表，
            初始化SSD网络模型，加载预训练权重，将模型移动到GPU设备（如果可用），设置图像均值，设置最大帧率，
            设置视频文件路径，初始化检测结果的列表，设置当前图像编号为0，设置录制路径，初始化情绪列表。

        """
        super(CameraShow, self).__init__(parent)
        self.setupUi(self)
        self.Timer = QTimer()
        self.timer = QTimer()
        self.time_first = time.time()
        self.time_ing = time.time()
        self.Timer.timeout.connect(self.show_img)
        self.timer.timeout.connect(self.showTime)
        self.PrepCamera()
        self.PrepareTorch()
        self.CallBackFunctions()
        self.showTime()
        self.case = 0
        self.frag_cap = True
        # self.Timer.timeout.connect(self.TimerOutFun)
        self.video_flg = True
        self.colors_tableau = [(214, 39, 40), (23, 190, 207), (188, 189, 34), (188, 34, 188), (205, 108, 8),
                               (150, 34, 188), (105, 108, 8)]
        # 初始化网络
        self.net = SSD()
        self.net = torch.nn.DataParallel(self.net)
        self.net.train(mode=False)
        # net.load_state_dict(torch.load('./weights/ssd300_VOC_100000.pth',map_location=lambda storage,loc: storage))
        # self.net.load_state_dict(
        #     torch.load('./weights/final_20200226_VOC_100000.pth', map_location=lambda storage, loc: storage))
        self.net.load_state_dict(torch.load('./weights/final_20200226_VOC_100000.pth',map_location=lambda storage, loc: storage.cuda(0)))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(device)
        if torch.cuda.is_available():
            self.net = self.net.cuda()
            cudnn.benchmark = True
        self.net = self.net.cuda()
        self.img_mean = (104.0, 117.0, 123.0)
        self.max_fps = 0

        # filename = ('D:\\大创\\Dataset\\3-FemaleGlasses.mp4')
        filename = ('D:\\大创\\Dangerous_driving_behavior_detection\\test\\3-FemaleGlasses.mp4')
        # 保存检测结果的List
        # 眼睛和嘴巴都是，张开为‘1’，闭合为‘0’

        self.Image_num = 0
        # self.RecordPath = 'E:/PythonEye/DachuangProject/test/3-FemaleGlasses.mp4'
        self.RecordPath = 'D:\\大创\\Dangerous_driving_behavior_detection\\test\\3-FemaleGlasses.mp4'
        self.VideoPath.setText(self.RecordPath)
        self.isRecordImg = False
        self.EMOTIONS = ["生气", "厌恶", "害怕", "喜悦", "悲伤", "惊讶", "普通"]
    #     prepare
    def PrepCamera(self):
        """
            准备相机函数，用于连接摄像头并显示相关信息。

            尝试连接摄像头，若成功则设置相关参数，清空消息框并显示提示信息“Oboard camera connected.”，更新当前时间；
            若连接失败，则清空消息框并显示错误信息。

            异常:
                Exception: 连接摄像头失败时引发的异常，显示相应错误信息。
        """
        try:
            self.camera = cv2.VideoCapture(0)
            self.Image_num = 0
            self.Msg.clear()
            self.Msg.append('Oboard camera connected.')
            self.Msg.setPlainText()
            self.showTime()
        except Exception as e:
            self.Msg.clear()
            self.Msg.append(str(e))

    def CallBackFunctions(self):
        """
            回调函数设置，用于连接按钮点击事件与相应的函数。

            连接按钮的点击事件与对应的功能函数，包括：
            - BtnRecord 按钮点击事件连接 setRecordImg 函数，用于设置记录图像；
            - btntestcamera 按钮点击事件连接 testCamera 函数，用于测试摄像头；
            - btn_start 按钮点击事件连接 StartDection 函数，用于启动检测；
            - btnexit 按钮点击事件连接 ExitApp 函数，用于退出应用程序；
            - btn_testvideo 按钮点击事件连接 testVideo 函数，用于测试视频；
            - BtnReadvideo 按钮点击事件连接 setFilePath 函数，用于设置文件路径。

        """
        self.BtnRecord.clicked.connect(self.setRecordImg)
        self.btntestcamera.clicked.connect(self.testCamera)
        # self.StopBt.clicked.connect(self.StopCamera)
        self.btn_start.clicked.connect(self.StartDection)
        self.btnexit.clicked.connect(self.ExitApp)
        self.btn_testvideo.clicked.connect(self.testVideo)
        self.BtnReadvideo.clicked.connect(self.setFilePath)

    # 显示时间
    def showTime(self):
        """
            显示当前时间。

            获取当前时间，并将其显示在界面上的 LCD 数字显示器中。

            函数逻辑包括：
            - 获取当前时间；
            - 启动定时器；
            - 将当前时间的小时、分钟和秒分别显示在 LCD 数字显示器中。

        """
        # time = QDateTime.currentDateTime()
        now_time = datetime.datetime.now()

        self.timer.start()
        # timeDisplay = time.toString("yyyy-MM-dd hh:mm:ss dddd")
        hour = now_time.strftime('%H')
        minute = now_time.strftime('%M')
        second = now_time.strftime('%S')
        self.TimeHourLCD.display(hour)
        self.TimeMinuteLCD.display(minute)
        self.TimeSecondLCD.display(second)

    def ColorAdjust(self, img):
        """
            调整图像的颜色通道。

            参数:
            - self: 包含此方法的类的实例。
            - img: 表示输入图像的numpy数组，形状为(高度, 宽度, 通道数)。

            返回:
            - img1: 表示颜色通道已调整的图像的numpy数组，形状为(高度, 宽度, 通道数)。

            异常:
            - Exception: 如果在颜色调整过程中发生错误。

            注意:
            此方法假定图像采用BGR格式（蓝色、绿色、红色），并将其重新排列为RGB格式（红色、绿色、蓝色）返回调整后的图像。
        """
        try:
            B = img[:, :, 0]
            G = img[:, :, 1]
            R = img[:, :, 2]
            img1 = img
            img1[:, :, 0] = B
            img1[:, :, 1] = G
            img1[:, :, 2] = R
            return img1
        except Exception as e:
            self.Msg.setPlainText(str(e))

    # 打开相机
    def testCamera(self):
        """
           测试相机功能。

           打开相机，并在界面上显示相机捕获的图像。如果相机未成功打开，则显示警告消息。

           函数逻辑包括：
           - 尝试打开相机；
           - 如果成功打开相机，则开始捕获图像并启动定时器以更新界面；
           - 如果相机已经打开，则关闭相机并停止定时器，清除显示的图像。

           注意:
           此方法依赖于类成员变量 camera、Timer、btntestcamera、btn_start、btn_testvideo、Image_num、timelb 和 Camera_2。
        """
        # self.camera = cv2.VideoCapture(0)
        print("测试相机")
        if self.Timer.isActive() == False:
            flag = self.camera.open(0)
            if flag == False:
                print("测试相机失败")
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                print("测试相机成功")
                self.case = 1
                self.timelb = time.clock()
                self.btntestcamera.setText(u'关闭相机')
                self.btn_start.setEnabled(False)
                self.btn_testvideo.setEnabled(False)
                self.Image_num = 0

                self.Timer.start(30)

        else:
            print("Timer.isActive()==true")
            self.Timer.stop()
            self.camera.release()
            # self.Camera.clear()
            self.Camera_2.clear()
            self.btntestcamera.setText(u'打开相机')
            self.btn_start.setEnabled(True)
            self.btn_testvideo.setEnabled(True)

    def PrepareTorch(self):
        """
            准备 PyTorch 环境。

            如果 CUDA 可用，则设置默认张量类型为 GPU 张量类型；否则设置为 CPU 张量类型。

            注意:
            此方法依赖于 torch 模块。
        """
        if torch.cuda.is_available():
            print('-----gpu mode-----')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            print('-----cpu mode-----')

    def TimerOutFun(self):
        """
            定时器超时函数。

            捕获相机图像，调整颜色通道，显示当前时间，并计算帧率。如果图像获取失败，则在消息框中显示错误信息。

            注意:
            此方法依赖于类成员变量 camera、Image、Image_num、timelb、Msg、ColorAdjust 方法和 showTime 方法。
        """
        success, img = self.camera.read()
        if success:
            self.Image = self.ColorAdjust(img)
            self.showTime()
            self.Image_num += 1
            if self.Image_num % 10 == 9:
                frame_rate = 10 / (time.clock() - self.timelb)
                self.FmRateLCD.display(frame_rate)
                self.timelb = time.clock()
        else:
            self.Msg.clear()
            self.Msg.setPlainText('Image obtaining failed.')

    def StartDection(self):
        """
           开始检测行为。

           打开相机并开始检测行为。检测包括眼部状态、嘴部状态、点头和摇头。如果相机未成功打开，则显示警告消息。

           注意:
           此方法依赖于类成员变量 camera、Timer、list_B、list_Y、list_Y1、blink_count、list_blink、danger_count、yawn_count、blink_freq、yawn_freq、timelb、btn_start、btntestcamera、btn_testvideo、open_t、danger_t、blink_start、yawn_start、danger_start、case、time_first、time_ing、point、nod_count、nod_freq、nod_fps、nod_start、shake_count、shake_freq、shake_start、shake_fps_l、shake_fps_r、FaceMaskTime 和 Facemasktemp。
        """

        if self.Timer.isActive() == False:
            flag = self.camera.open(0)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:

                self.list_B = np.ones(10)  # 眼睛状态List,建议根据fps修改
                self.list_Y = np.zeros(50)  # 嘴巴状态list，建议根据fps修改
                self.list_Y1 = np.ones(8)  # 如果在list_Y中存在list_Y1，则判定一次打哈欠，同上，长度建议修改
                self.blink_count = 0  # 眨眼计数
                self.list_blink=np.ones(60)  #判断60帧的睁眼闭眼
                self.danger_count=0#危险行为帧
                self.yawn_count = 0#哈欠帧
                self.blink_freq = 0.5
                self.yawn_freq = 0
                self.timelb = time.clock()
                self.btn_start.setText(u'停止运行')
                self.btntestcamera.setEnabled(False)
                self.btn_testvideo.setEnabled(False)
                self.open_t = 0  # 用于刷新眼部状态label
                self.danger_t=0 #用于刷新危险行为状态
                self.blink_start = time.time()  # 眨眼时间
                self.yawn_start = time.time()  # 打哈欠时间
                self.danger_start=time.time()#吸烟or打电话时间
                self.case = 2
                self.Timer.start(30)
                self.time_first = time.time()
                self.time_ing = time.time()
                self.point = []
                self.nod_count=0 #点头次数
                self.nod_freq = 0  # 点头频率
                self.nod_fps=0 #点头帧数
                self.nod_start=time.time()#点头时间
                self.shake_count = 0  # 摇头次数
                self.shake_freq=0 #摇头频率
                self.shake_start=time.time()#摇头时间
                self.shake_fps_l=0 #摇头帧数
                self.shake_fps_r = 0 # 摇头帧数

                # 口罩检测时间
                self.FaceMaskTime=time.time()
                self.Facemasktemp=1

        else:
            self.Timer.stop()
            self.camera.release()

            self.Camera_2.clear()
            self.btn_start.setText(u'开始运行')
            self.btn_testvideo.setEnabled(True)
            self.btntestcamera.setEnabled(True)

    def show_img(self):
        """
            显示摄像头捕获的图像。

            捕获图像后进行各种行为检测，包括面部识别、口罩检测、点头、摇头和表情识别。

            注意:
            此方法依赖于全局变量 temp_t 和类成员变量 camera、img、Image_num、timelb、case、Camera_2、frag_cap、point、FaceMaskTime、Facemasktemp、nod_fps、nod_count、Head_state、nod_start、nod_freq、shake_fps_l、shake_fps_r、shake_count、shake_freq、shake_start、Emotion、Emotion_pred 和 label_pred_img。
        """
        global temp_t
        self.camera = cv2.VideoCapture(0)
        success, self.img = self.camera.read()
        if success:
            self.Image_num += 1
            if self.Image_num % 10 == 9:
                frame_rate = 10 / (time.clock() - self.timelb)
                self.FmRateLCD.display(frame_rate)
                self.timelb = time.clock()
            if self.case == 0:
                showImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                showImg = qimage2ndarray.array2qimage(showImg)
                self.Camera_2.setPixmap(QPixmap(showImg))  # 展示图片
                self.Camera_2.show()
            if self.case == 1:
                bounding_boxes, landmarks = detect_faces(self.img)
                self.img = show_bboxes(self.img, bounding_boxes, landmarks)
                showImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                showImg = qimage2ndarray.array2qimage(showImg)
                self.Camera_2.setPixmap(QPixmap(showImg))  # 展示图片
                self.Camera_2.show()
            if self.case == 2:
                print("case:2")
                img_copy = self.img.copy()
                frag_gray = False
                self.time_ing = time.time()
                # point=[100,0,540,480]
                if self.frag_cap:
                    bounding_boxes, landmarks = detect_faces(self.img)
                    print('正在定位······')
                    if len(bounding_boxes)== 1:
                        self.point.clear()
                        for b in bounding_boxes:
                            b = [int(round(value)) for value in b]
                            for i in b:
                                self.point.append(i)
                        self.frag_cap = False
                    # print(point)
                    # cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                    # 裁剪坐标为[y0:y1, x0:x1]

                if not self.frag_cap:
                    if self.point[0] < 540:
                        self.img = self.img[self.point[1] - 10:479, self.point[0] - 100:self.point[2] + 100]
                    else:
                        self.img = self.img[self.point[1] - 10:479, self.point[0] - 100:639]
                else:
                    self.img = self.img[1:479, 1:640]
                if int(self.time_ing - self.time_first) % 60 == 0:
                    self.frag_cap = True

                else:
                    self.frag_cap = False
                bounding_boxes, landmarks = detect_faces(self.img)


                # 定时进行口罩检测

                if time.time() -self.FaceMaskTime>self.Facemasktemp:
                    self.Facemasktemp = 60
                    if len(bounding_boxes)>0:
                        self.FaceMaskTime = time.time()
                        label=get_face_state(get_face_expression(self.img, bounding_boxes))
                        if label=='nomask':
                            self.Msg.setText("为了您的安全，请佩戴口罩！")

                        else:
                            self.Msg.setText("祝你旅途愉快！")


                #通过MTCNN人脸框判断，当检测不到人脸时判断低头or瞌睡
                if len(bounding_boxes) == 0:
                    self.nod_fps+=1
                if self.nod_fps>=3:
                    self.Head_state.setText('点头')
                    self.nod_count+=1
                if len(bounding_boxes) > 0:
                    self.nod_fps=0

                #通过头部姿态欧拉角角度变化判断是否摇头
                if len(bounding_boxes) > 0:
                    Head_Y_X_Z = get_head_pose(landmarks)
                    # print('pitch:{}, yaw:{}, roll:{}'.format(Head_Y_X_Z[1], Head_Y_X_Z[2], Head_Y_X_Z[3]))
                    if(Head_Y_X_Z[2]<-0.75):
                        self.shake_fps_l+=1
                    if(Head_Y_X_Z[2]>=-0.75):
                        self.shake_fps_l = 0
                    if self.shake_fps_l>=5:
                        self.shake_count+=1
                        self.Head_state.setText('摇头')
                    if Head_Y_X_Z[3]>=0.30:
                        self.shake_fps_r+=1
                    if self.shake_fps_r>=5:
                        self.shake_count+=1
                        self.Head_state.setText('摇头')
                    if Head_Y_X_Z[3]<0.30:
                        self.shake_fps_r=0
                    # print(Head_Y_X_Z[1])
                    # print(Head_Y_X_Z[2])
                    # print(Head_Y_X_Z[3])

                if time.time() - self.nod_start > 3:
                    self.Head_state.setText('')
                if time.time() - self.shake_start > 3:
                    self.Head_state.setText('')
                # 计算低头频率  每10s计算一次
                if time.time() - self.nod_start > 10:
                    times = time.time() - self.nod_start
                    self.nod_freq = self.nod_count / times
                    self.nod_start = time.time()
                    self.Nod_LCD.display(self.nod_freq)


                # 计算摇头频率
                if time.time() - self.shake_start > 10:
                    times = time.time() - self.shake_start
                    self.shake_freq = self.shake_count / times
                    self.shake_start = time.time()
                    self.shake_LCD.display(self.shake_freq)




                if len(bounding_boxes)>0:
                    Emotions = get_emotion(get_face_expression(self.img, bounding_boxes))
                    self.Emotion.setText(Emotions[1])
                    self.Emotion_pred.display(float(Emotions[0]))
                    # print(Emotions)

                    canvas = cv2.imread('img_resource/label_pred.jpg', flags=cv2.IMREAD_UNCHANGED)
                    for (i, (emotion, prob)) in enumerate(zip(self.EMOTIONS, Emotions[2])):
                        # text = "{}: {:.2f}%".format(emotion, prob * 100)
                        text = "{:.2f}%".format(prob * 100)
                        # 绘制表情类和对应概率的条形图
                        w = int(prob * 180)
                        # print(text)
                        # canvas = 255 * np.ones((250, 300, 3), dtype="uint8")

                        cv2.rectangle(canvas, (0, (i * 44) + 25), (w, (i * 43) + 40), (100, 200, 130), -1)
                        cv2.putText(canvas, text, (170, (i * 43) + 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                        show = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
                        # cv2.imshow('test', showImage)
                        # showImg=QPixmap(showImage)
                        self.label_pred_img.setPixmap(QtGui.QPixmap.fromImage(showImage))
                #         # print('test')
                # print('Head_Y_X_Z')
                # print(Head_Y_X_Z)

                x = cv2.resize(self.img, (300, 300)).astype(np.float32)
                flag_B = True  # 是否闭眼的flag
                flag_Y = False
                num_rec = 0  # 检测到的眼睛的数量

                # 分界线
                x -= self.img_mean
                x = x.astype(np.float32)
                x = x[:, :, ::-1].copy()
                x = torch.from_numpy(x).permute(2, 0, 1)
                xx = Variable(x.unsqueeze(0))
                # if torch.cuda.is_available():
                #     xx = xx.cuda()
                xx = xx.cuda()
                y = self.net(xx)
                softmax = nn.Softmax(dim=-1)
                detect = Detect(config.class_num, 0, 200, 0.01, 0.45)
                priors = utils.default_prior_box()

                loc, conf = y
                loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
                conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

                detections = detect(
                    loc.view(loc.size(0), -1, 4),
                    softmax(conf.view(conf.size(0), -1, config.class_num)),
                    torch.cat([o.view(-1, 4) for o in priors], 0)
                ).data
                labels = VOC_CLASSES
                # 将检测结果放置于图片上
                scale = torch.Tensor(self.img.shape[1::-1]).repeat(2)
                self.img = show_bboxes(self.img, bounding_boxes, landmarks)
                for i in range(detections.size(1)):

                    j = 0
                    while detections[0, i, j, 0] >= 0.4:
                        score = detections[0, i, j, 0]
                        label_name = labels[i - 1]
                        if label_name == 'calling' and score > 0.8:
                            self.Danger_state.setText('打电话')
                            self.danger_count += 1
                            frag_gray = True
                        if label_name == 'smoke' and score > 0.8:
                            self.Danger_state.setText('吸烟')
                            self.danger_count += 1
                            frag_gray = True
                        if label_name!='smoke'and label_name!='calling':
                             self.danger_t+=1
                        if self.danger_t>=20:
                            self.Danger_state.setText('')
                            self.danger_t=0
                        if label_name == 'open_eye':
                            self.open_t += 1
                            if self.open_t >= 20:
                                self.Eyes_state.setText('')
                                self.open_t = 0
                        if label_name == 'closed_mouth':
                            self.Mouth_state.setText(' ')
                        if label_name == 'closed_eye':
                            flag_B = False
                            frag_gray = True
                        if label_name == 'open_mouth':
                            flag_Y = True
                        display_txt = '%s:%.2f' % (label_name, score)
                        pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                        self.coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
                        color = self.colors_tableau[i]
                        cv2.rectangle(self.img, (pt[0], pt[1]), (pt[2], pt[3]), color, 2)
                        cv2.putText(self.img, display_txt, (int(pt[0]), int(pt[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (255, 255, 255),
                                    1, 8)
                        j += 1
                        num_rec += 1

                # cv2.imshow('test', self.img)
                if num_rec > 0:
                    if flag_B:
                        # print(' 1:eye-open')
                        self.list_B = np.append(self.list_B, 1)  # 睁眼为‘1’
                        self.list_blink=np.append(self.list_blink,1)
                    else:
                        # print(' 0:eye-closed')

                        self.list_B = np.append(self.list_B, 0)  # 闭眼为‘0’
                        self.list_blink = np.append(self.list_blink, 0)
                    self.list_blink = np.delete(self.list_blink, 0)
                    self.list_B = np.delete(self.list_B, 0)
                    if flag_Y:
                        self.list_Y = np.append(self.list_Y, 1)
                    else:
                        self.list_Y = np.append(self.list_Y, 0)
                    self.list_Y = np.delete(self.list_Y, 0)
                else:
                    self.Msg.clear()
                    self.Msg.setPlainText('Nothing detected.')

                # print(list)
                # 实时计算PERCLOS
                self.perclos = 1 - np.average(self.list_blink)
                # print('perclos={:f}'.format(perclos))
                self.PERCLOS.display(self.perclos)
                if self.list_B[8] == 1 and self.list_B[9] == 0:
                    # 如果上一帧为’1‘，此帧为’0‘则判定为眨眼
                    self.Eyes_state.setText('眨眼')
                    self.blink_count += 1
                    frag_gray = True
                    str = datetime.datetime.now().strftime("%H:%M:%S")
                    self.State_record.append(str + '：眨眼')
                    # img_copy=cv2.cvtColor(img_copy,cv2.COLOR_RGB2GRAY)
                blink_T = time.time() - self.blink_start
                if blink_T > 30:
                    # 每30秒计算一次眨眼频率
                    blink_freq = self.blink_count / blink_T
                    self.blink_start = time.time()
                    self.blink_count = 0
                    print('blink_freq={:f}'.format(blink_freq))
                    self.Blink_freq.display(blink_freq * 2)
                # 检测打哈欠
                # if Yawn(list_Y,list_Y1):
                if (self.list_Y[len(self.list_Y) - len(self.list_Y1):] == self.list_Y1).all():
                    # print('----------------------打哈欠----------------------')
                    self.Mouth_state.setText('打哈欠')
                    self.yawn_count += 1
                    frag_gray = True
                    str = datetime.datetime.now().strftime("%H:%M:%S")
                    self.State_record.append(str + '：打哈欠')
                    self.list_Y = np.zeros(50)
                # 计算打哈欠频率
                yawn_T = time.time() - self.yawn_start
                if yawn_T > 60:
                    yawn_freq = self.yawn_count / yawn_T
                    self.yawn_start = time.time()
                    self.yawn_count = 0
                    print('yawn_freq={:f}'.format(yawn_freq))
                    self.Yawn_freq.display(yawn_freq)






                # 计算危险行为频率
                DangerAct_T = time.time() - self.danger_start
                if DangerAct_T > 60:
                    danger_freq = self.danger_count / DangerAct_T
                    self.danger_start = time.time()
                    self.danger_count = 0
                    print('danger_freq={:f}'.format(danger_freq))
                    self.Danger_LCD.display(danger_freq)

                if (self.perclos > 0.4):
                    # print('疲劳')
                    self.State.setText('疲劳')
                elif (self.blink_freq > 0.3):
                    # print('疲劳')
                    self.State.setText('疲劳')
                    self.blink_freq = 0  # 如果因为眨眼频率判断疲劳，则初始化眨眼频率
                elif (self.yawn_freq > 5.0 / 60):
                    # print("疲劳")
                    self.State.setText('疲劳')
                    self.yawn_freq = 0  # 初始化，同上
                else:
                    self.State.setText('清醒')

                if not frag_gray:
                    showImg = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
                else:
                    if self.isRecordImg:
                        str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                        temp = 'ImgRecord/' + str + '.jpg'
                        cv2.imwrite(temp, img_copy)

                    showImg = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
                showImg = qimage2ndarray.array2qimage(showImg)
                self.Camera_2.setPixmap(QPixmap(showImg))  # 展示图片
                self.Camera_2.show()
            if self.case == 3:
                img_copy = self.img.copy()
                frag_gray = False
                self.time_ing = time.time()
                # point=[100,0,540,480]
                if self.frag_cap:
                    bounding_boxes, landmarks = detect_faces(self.img)
                    print('正在定位······')
                    if len(bounding_boxes) == 1:
                        self.point.clear()
                        for b in bounding_boxes:
                            b = [int(round(value)) for value in b]
                            for i in b:
                                self.point.append(i)
                        self.frag_cap = False
                    # print(point)
                    # cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                    # 裁剪坐标为[y0:y1, x0:x1]

                if not self.frag_cap:
                    if self.point[0] < 540:
                        self.img = self.img[self.point[1] - 10:479, self.point[0] - 100:self.point[2] + 100]
                    else:
                        self.img = self.img[self.point[1] - 10:479, self.point[0] - 100:639]
                else:
                    self.img = self.img[1:479, 1:640]
                if int(self.time_ing - self.time_first) % 60 == 0:
                    self.frag_cap = True

                else:
                    self.frag_cap = False
                bounding_boxes, landmarks = detect_faces(self.img)

                # 定时进行口罩检测

                if time.time() - self.FaceMaskTime > self.Facemasktemp:
                    self.Facemasktemp = 60
                    if len(bounding_boxes) > 0:
                        label = get_face_state(get_face_expression(self.img, bounding_boxes))
                        self.FaceMaskTime=time.time()
                        if label == 'nomask':
                            self.Msg.setText("为了您的安全，请佩戴口罩！")
                        else:
                            self.Msg.setText("祝你旅途愉快！")






                # print(get_face_state(self.img, bounding_boxes))
                # 通过MTCNN人脸框判断，当检测不到人脸时判断低头or瞌睡
                if len(bounding_boxes) == 0:
                    self.nod_fps += 1
                if self.nod_fps >= 3:
                    self.Head_state.setText('点头')
                    self.nod_count += 1
                if len(bounding_boxes) > 0:
                    self.nod_fps = 0

                # 通过头部姿态欧拉角角度变化判断是否摇头
                if len(bounding_boxes) > 0:
                    Head_Y_X_Z = get_head_pose(landmarks)
                    # print('pitch:{}, yaw:{}, roll:{}'.format(Head_Y_X_Z[1], Head_Y_X_Z[2], Head_Y_X_Z[3]))
                    if (Head_Y_X_Z[2] < -0.75):
                        self.shake_fps_l += 1
                    if (Head_Y_X_Z[2] >= -0.75):
                        self.shake_fps_l = 0
                    if self.shake_fps_l >= 5:
                        self.shake_count += 1
                        self.Head_state.setText('摇头')
                    if Head_Y_X_Z[3] >= 0.30:
                        self.shake_fps_r += 1
                    if self.shake_fps_r >= 5:
                        self.shake_count += 1
                        self.Head_state.setText('摇头')
                    if Head_Y_X_Z[3] < 0.30:
                        self.shake_fps_r = 0
                    # print(Head_Y_X_Z[1])
                    # print(Head_Y_X_Z[2])
                    # print(Head_Y_X_Z[3])

                if time.time() - self.nod_start > 3:
                    self.Head_state.setText('')
                if time.time() - self.shake_start > 3:
                    self.Head_state.setText('')
                # 计算低头频率  每10s计算一次
                if time.time() - self.nod_start > 10:
                    times = time.time() - self.nod_start
                    self.nod_freq = self.nod_count / times
                    self.nod_start = time.time()
                    self.Nod_LCD.display(self.nod_freq)

                # 计算摇头频率
                if time.time() - self.shake_start > 10:
                    times = time.time() - self.shake_start
                    self.shake_freq = self.shake_count / times
                    self.shake_start = time.time()
                    self.shake_LCD.display(self.shake_freq)

                if len(bounding_boxes) > 0:

                    Emotions = get_emotion(get_face_expression(self.img, bounding_boxes))
                    self.Emotion.setText(Emotions[1])
                    self.Emotion_pred.display(float(Emotions[0]))
                    # print(Emotions)
                    # print(get_face_state(get_face_expression(self.img, bounding_boxes)))
                    canvas = cv2.imread('img_resource/label_pred.jpg', flags=cv2.IMREAD_UNCHANGED)
                    for (i, (emotion, prob)) in enumerate(zip(self.EMOTIONS, Emotions[2])):
                        # text = "{}: {:.2f}%".format(emotion, prob * 100)
                        text = "{:.2f}%".format(prob * 100)
                        # 绘制表情类和对应概率的条形图
                        w = int(prob * 180)
                        # print(text)
                        # canvas = 255 * np.ones((250, 300, 3), dtype="uint8")

                        cv2.rectangle(canvas, (0, (i * 44) + 25), (w, (i * 43) + 40), (100, 200, 130), -1)
                        cv2.putText(canvas, text, (170, (i * 43) + 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                        show = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
                        # cv2.imshow('test', showImage)
                        # showImg=QPixmap(showImage)
                        self.label_pred_img.setPixmap(QtGui.QPixmap.fromImage(showImage))
                #         # print('test')
                # print('Head_Y_X_Z')
                # print(Head_Y_X_Z)

                x = cv2.resize(self.img, (300, 300)).astype(np.float32)
                flag_B = True  # 是否闭眼的flag
                flag_Y = False
                num_rec = 0  # 检测到的眼睛的数量

                # 分界线
                x -= self.img_mean
                x = x.astype(np.float32)
                x = x[:, :, ::-1].copy()
                x = torch.from_numpy(x).permute(2, 0, 1)
                xx = Variable(x.unsqueeze(0))
                # if torch.cuda.is_available():
                #     xx = xx.cuda()
                xx = xx.cuda()
                y = self.net(xx)
                softmax = nn.Softmax(dim=-1)
                detect = Detect(config.class_num, 0, 200, 0.01, 0.45)
                priors = utils.default_prior_box()

                loc, conf = y
                loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
                conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

                detections = detect(
                    loc.view(loc.size(0), -1, 4),
                    softmax(conf.view(conf.size(0), -1, config.class_num)),
                    torch.cat([o.view(-1, 4) for o in priors], 0)
                ).data
                labels = VOC_CLASSES
                # 将检测结果放置于图片上
                scale = torch.Tensor(self.img.shape[1::-1]).repeat(2)
                self.img = show_bboxes(self.img, bounding_boxes, landmarks)
                for i in range(detections.size(1)):

                    j = 0
                    while detections[0, i, j, 0] >= 0.4:
                        score = detections[0, i, j, 0]
                        label_name = labels[i - 1]
                        if label_name == 'calling' and score > 0.8:
                            self.Danger_state.setText('打电话')
                            self.danger_count += 1
                            frag_gray = True
                        if label_name == 'smoke' and score > 0.8:
                            self.Danger_state.setText('吸烟')
                            self.danger_count += 1
                            frag_gray = True
                        if label_name != 'smoke' and label_name != 'calling':
                            self.danger_t += 1
                        if self.danger_t >= 20:
                            self.Danger_state.setText('')
                            self.danger_t = 0
                        if label_name == 'open_eye':
                            self.open_t += 1
                            if self.open_t >= 20:
                                self.Eyes_state.setText('')
                                self.open_t = 0
                        if label_name == 'closed_mouth':
                            self.Mouth_state.setText(' ')
                        if label_name == 'closed_eye':
                            flag_B = False
                            frag_gray = True
                        if label_name == 'open_mouth':
                            flag_Y = True
                        display_txt = '%s:%.2f' % (label_name, score)
                        pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                        self.coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
                        color = self.colors_tableau[i]
                        cv2.rectangle(self.img, (pt[0], pt[1]), (pt[2], pt[3]), color, 2)
                        cv2.putText(self.img, display_txt, (int(pt[0]), int(pt[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (255, 255, 255),
                                    1, 8)
                        j += 1
                        num_rec += 1

                # cv2.imshow('test', self.img)
                if num_rec > 0:
                    if flag_B:
                        # print(' 1:eye-open')
                        self.list_B = np.append(self.list_B, 1)  # 睁眼为‘1’
                        self.list_blink = np.append(self.list_blink, 1)
                    else:
                        # print(' 0:eye-closed')

                        self.list_B = np.append(self.list_B, 0)  # 闭眼为‘0’
                        self.list_blink = np.append(self.list_blink, 0)
                    self.list_blink = np.delete(self.list_blink, 0)
                    self.list_B = np.delete(self.list_B, 0)
                    if flag_Y:
                        self.list_Y = np.append(self.list_Y, 1)
                    else:
                        self.list_Y = np.append(self.list_Y, 0)
                    self.list_Y = np.delete(self.list_Y, 0)
                else:
                    self.Msg.clear()
                    # self.Msg.setPlainText('Nothing detected.')

                # print(list)
                # 实时计算PERCLOS
                self.perclos = 1 - np.average(self.list_blink)
                # print('perclos={:f}'.format(perclos))
                self.PERCLOS.display(self.perclos)
                if self.list_B[8] == 1 and self.list_B[9] == 0:
                    # 如果上一帧为’1‘，此帧为’0‘则判定为眨眼
                    self.Eyes_state.setText('眨眼')
                    self.blink_count += 1
                    frag_gray = True
                    str = datetime.datetime.now().strftime("%H:%M:%S")
                    self.State_record.append(str + '：眨眼')
                    # img_copy=cv2.cvtColor(img_copy,cv2.COLOR_RGB2GRAY)
                blink_T = time.time() - self.blink_start
                if blink_T > 30:
                    # 每30秒计算一次眨眼频率
                    blink_freq = self.blink_count / blink_T
                    self.blink_start = time.time()
                    self.blink_count = 0
                    print('blink_freq={:f}'.format(blink_freq))
                    self.Blink_freq.display(blink_freq * 2)
                # 检测打哈欠
                # if Yawn(list_Y,list_Y1):
                if (self.list_Y[len(self.list_Y) - len(self.list_Y1):] == self.list_Y1).all():
                    # print('----------------------打哈欠----------------------')
                    self.Mouth_state.setText('打哈欠')
                    self.yawn_count += 1
                    frag_gray = True
                    str = datetime.datetime.now().strftime("%H:%M:%S")
                    self.State_record.append(str + '：打哈欠')
                    self.list_Y = np.zeros(50)
                # 计算打哈欠频率
                yawn_T = time.time() - self.yawn_start
                if yawn_T > 60:
                    yawn_freq = self.yawn_count / yawn_T
                    self.yawn_start = time.time()
                    self.yawn_count = 0
                    print('yawn_freq={:f}'.format(yawn_freq))
                    self.Yawn_freq.display(yawn_freq)

                # 计算危险行为频率
                DangerAct_T = time.time() - self.danger_start
                if DangerAct_T > 60:
                    danger_freq = self.danger_count / DangerAct_T
                    self.danger_start = time.time()
                    self.danger_count = 0
                    print('danger_freq={:f}'.format(danger_freq))
                    self.Danger_LCD.display(danger_freq)

                if (self.perclos > 0.4):
                    # print('疲劳')
                    self.State.setText('疲劳')
                elif (self.blink_freq > 0.3):
                    # print('疲劳')
                    self.State.setText('疲劳')
                    self.blink_freq = 0  # 如果因为眨眼频率判断疲劳，则初始化眨眼频率
                elif (self.yawn_freq > 5.0 / 60):
                    # print("疲劳")
                    self.State.setText('疲劳')
                    self.yawn_freq = 0  # 初始化，同上
                else:
                    self.State.setText('清醒')

                if not frag_gray:
                    showImg = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
                else:
                    if self.isRecordImg:
                        str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                        temp = 'ImgRecord/' + str + '.jpg'
                        cv2.imwrite(temp, img_copy)

                    showImg = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
                self.State_record.moveCursor(QTextCursor.End)
                showImg = qimage2ndarray.array2qimage(showImg)
                self.Camera_2.setPixmap(QPixmap(showImg))  # 展示图片
                self.Camera_2.show()

    #测试视频
    def testVideo(self):
        """
            测试视频函数。

            当定时器未启动时，打开摄像头并开始视频测试。如果摄像头打开失败，则显示警告消息。如果成功打开摄像头，则初始化各种参数和计时器，并更新界面按钮状态。

            当定时器已经启动时，停止视频测试并释放摄像头资源，同时恢复界面按钮状态。

            Parameters:
                self (object): 类实例对象。

            Returns:
                None
        """
        if self.Timer.isActive() == False:
            flag = self.camera.open(0)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                path=self.RecordPath
                self.camera = cv2.VideoCapture(path)
                self.list_B = np.ones(10)  # 眼睛状态List,建议根据fps修改
                self.list_Y = np.zeros(50)  # 嘴巴状态list，建议根据fps修改
                self.list_Y1 = np.ones(8)  # 如果在list_Y中存在list_Y1，则判定一次打哈欠，同上，长度建议修改
                self.blink_count = 0  # 眨眼计数
                self.list_blink = np.ones(60)  # 判断60帧的睁眼闭眼
                self.danger_count = 0  # 危险行为帧
                self.yawn_count = 0  # 哈欠帧
                self.blink_freq = 0.5
                self.yawn_freq = 0
                self.timelb = time.clock()
                self.btn_testvideo.setText(u'停止测试')
                self.btntestcamera.setEnabled(False)
                self.btn_start.setEnabled(False)
                self.open_t = 0  # 用于刷新眼部状态label
                self.danger_t = 0  # 用于刷新危险行为状态
                self.blink_start = time.time()  # 眨眼时间
                self.yawn_start = time.time()  # 打哈欠时间
                self.danger_start = time.time()  # 吸烟or打电话时间
                self.case = 3
                self.Timer.start(30)
                self.time_first = time.time()
                self.time_ing = time.time()
                self.point = []
                self.nod_count = 0  # 点头次数
                self.nod_freq = 0  # 点头频率
                self.nod_fps = 0  # 点头帧数
                self.nod_start = time.time()  # 点头时间
                self.shake_count = 0  # 摇头次数
                self.shake_freq = 0  # 摇头频率
                self.shake_start = time.time()  # 摇头时间
                self.shake_fps_l = 0  # 摇头帧数
                self.shake_fps_r = 0  # 摇头帧数

                # 口罩检测时间
                self.FaceMaskTime = time.time()
                self.Facemasktemp = 1
        else:
            self.Timer.stop()
            self.camera.release()
            self.camera=cv2.VideoCapture(0)
            self.Camera_2.clear()
            self.btn_testvideo.setText(u'测试视频')
            self.btn_start.setEnabled(True)
            self.btntestcamera.setEnabled(True)

    # 退出程序
    def ExitApp(self, event):
        """
           退出应用程序函数。

           当用户尝试关闭应用程序时触发，显示一个消息框询问用户是否确定关闭。如果用户选择取消，则忽略关闭事件；如果用户选择确定，则释放摄像头资源并停止定时器，并接受关闭事件。

           Parameters:
               event (object): 事件对象，用于接收关闭事件。

           Returns:
               None
        """
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cacel.setText(u'取消')
        # msg.setDetailedText('sdfsdff')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            #             self.socket_client.send_command(self.socket_client.current_user_command)
            if self.camera.isOpened():
                self.camera.release()
            if self.Timer.isActive():
                self.Timer.stop()
            event.accept()

# 是否记录
    def setRecordImg(self):
        """
           设置记录图像函数。

           当用户点击按钮时触发，用于切换记录图像的状态。如果当前状态为“记录图像”，则将按钮文本设置为“停止记录”并将isRecordImg标志设置为True；如果当前状态为“停止记录”，则将按钮文本设置为“记录图像”并将isRecordImg标志设置为False。

           Returns:
               None
        """
        tag=self.BtnRecord.text()
        if tag=='记录图像':
            self.BtnRecord.setText('停止记录')
            self.isRecordImg=True
        elif tag=='停止记录':
            self.BtnRecord.setText('记录图像')
            self.isRecordImg=False


#文件路径
    def setFilePath(self):
        """
           设置文件路径函数。

           用户点击按钮时触发，用于选择文件路径并将选择的文件路径设置为RecordPath，并更新显示在VideoPath标签上。

           Returns:
               None
        """
        # dirname = QFileDialog.getExistingDirectory(self, "浏览", '.*')
        fileName1, filetype = QFileDialog.getOpenFileName(self,
                                                          "选取文件")
        if fileName1:
            self.RecordPath = fileName1
            self.VideoPath.setText(self.RecordPath)

    #     关闭 X
    def closeEvent(self, event):
        """
           关闭事件处理函数。

           当用户尝试关闭窗口时触发，显示询问是否关闭的消息框。如果用户选择取消，则忽略事件；如果用户选择确定，停止摄像头和计时器，并接受关闭事件。

           Args:
               event: 事件对象

           Returns:
               None
        """
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cacel.setText(u'取消')
        # msg.setDetailedText('sdfsdff')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            #             self.socket_client.send_command(self.socket_client.current_user_command)
            if self.camera.isOpened():
                self.camera.release()
            if self.Timer.isActive():
                self.Timer.stop()
            event.accept()


if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    app = QApplication(sys.argv)
    ui = CameraShow()
    print("测试相机成功")
    ui.show()
    sys.exit(app.exec_())
