import tkinter as tk
from PIL import Image,ImageTk
import cv2
import torch
import matplotlib.pyplot  as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from statistics import mode
import sys
from models import get_model 
from ema_pytorch import EMA
import torch.nn.functional as F
from torchvision import transforms
from tkinter import filedialog
from tkinter import messagebox
from residual import Residual, GlobalAvgPool2d, ResNet, resnet_block
from moviepy.editor import VideoFileClip
import torch.nn as nn
from torch.utils.data import DataLoader
# 定义模型
model = get_model(num_classes=7, model_size="base")  # 替换为实际的类别数
ema = EMA(model, beta=0.99, update_every=16)
# 加载模型的状态字典
model_path = 'model/EmoNeXt_base.pkl'  # 替换为你的模型文件路径
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
# 提取模型的状态字典
if 'model' in checkpoint:
    state_dict = checkpoint['model']
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()
device = torch.device("cpu")
# 定义输入图像的预处理
transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(236),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    )

resnet = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7 , stride=2, padding=3),
    nn.BatchNorm2d(64),                  #归一化
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
resnet.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
resnet.add_module("resnet_block2", resnet_block(64, 128, 2))
resnet.add_module("resnet_block3", resnet_block(128, 256, 2))
resnet.add_module("resnet_block4", resnet_block(256, 512, 2))
resnet.add_module("global_avg_pool", GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
resnet.add_module("fc", nn.Sequential(ResNet(), nn.Linear(512, 7)))


#opencv自带的一个面部识别分类器
detection_model_path = 'model/haarcascade_frontalface_default.xml'

# 加载人脸检测模型
face_detection = cv2.CascadeClassifier(detection_model_path)
# 表情标签
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

picture_path=""

emotion_history=[]

def output(face):
    face = Image.fromarray(face)
    # 保存图像到文件
    crops = transform(face)  
    crops = crops.unsqueeze(0)
    predictions, a = model(crops)
    probabilities = F.softmax(a, dim=1)
    return probabilities.detach().numpy()

def preprocess_input(images):
    images = images/255.0
    return images

def windows():
    def picture():
        window_1=tk.Tk()
        window_1.title("图片表情识别")
        window_1.geometry('1000x600')
        tk.Label(window_1,text="图片表情识别",font=("黑体",28),
                width=30,
                height=2).place(x=500,y=20)

        filepath = filedialog.askopenfilename(
            title='指定需要识别的图片',
            filetypes=[('All files', '*')]
        )
        
        image_path = filepath
        frame = cv2.imread(image_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        # 获取全部人脸
        faces = face_detection.detectMultiScale(gray, 1.3, 5)
        if not isinstance(faces, np.ndarray):
            if faces == ():
                faces = np.array([[0, 0, height, width]])
        # 对于所有发现的人脸
        for (x, y, w, h) in faces:
            # 在脸周围画一个矩形框，(255,0,0)是颜色，2是线宽
            cv2.rectangle(frame, (x, y), (x + w, y + h), (84, 255, 159), 2)
            # 获取人脸图像
            face = gray[y:y + h, x:x + w]
            emotion_arg_predict = output(face)  # MARK:保存的是几种表情的预测值的大小。.
            emotion_arg = np.argmax(emotion_arg_predict)
            emotion = emotion_labels[emotion_arg]
            emotion_arg_predict = emotion_arg_predict.tolist()[0]
            arg_max=max(emotion_arg_predict)
            arg_min=min(emotion_arg_predict)
            fenmu=arg_max-arg_min
            emotion_arg_predict = [( i-arg_min ) / fenmu for i in emotion_arg_predict]
            # 在矩形框上部，输出分类文字
            cv2.putText(frame, emotion, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 255), 1, cv2.LINE_AA)
        # 显示结果图片
        cv2.namedWindow("Emotion Recognition", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv.namedWindow("input", 0)
        cv2.resizeWindow("Emotion Recognition", 400, 300)
        cv2.imshow('Emotion Recognition', frame)
        # f = Figure(figsize=(5, 4), dpi=100)
        # a = f.add_subplot(111)  # 添加子图:1行1列第1个
        plt.figure(figsize=(650, 6))
        fig, axs = plt.subplots(1, 2, sharey=False)

        # bar_label = emotions.values()

        # axs[0].imshow(test_image_array[image_number], 'gray')
        axs[0].imshow(gray, 'gray')
        axs[0].set_title(emotion)
        #
        print(list(emotion_labels.values()))
        axs[1].bar(list(emotion_labels.values()), emotion_arg_predict,width=0.5, color='orange', alpha=0.7)
        axs[1].grid()
        canvas = FigureCanvasTkAgg(fig, master=window_1)

        canvas.draw()  # 注意show方法已经过时了,这里改用draw
        canvas.get_tk_widget().pack(side=tk.TOP,  # 上对齐
                                    fill=tk.BOTH,  # 填充方式
                                    expand=tk.YES)  # 随窗口大小调整而调整
        toolbar = NavigationToolbar2Tk(canvas, window_1)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def video():
        emotion_classifier = torch.load("model/model_resnet_final.pkl")
        video_path = tk.filedialog.askopenfilename(parent=window, title='指定需要识别的视频',
                                                 filetypes=[('All files', '*')])  # 从文件中打开，打开方式
        frame_window = 15

        emotion_window = []
        emotion_history= []


        
        video_capture = cv2.VideoCapture(video_path)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.startWindowThread()
        cv2.namedWindow('video face recognition')
        # 获取视频帧率和尺寸
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = video_path.replace('.mp4', '_output.mp4')

        # 创建视频写入对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        while True:
            # 读取一帧
            _, frame = video_capture.read()
            frame = frame.copy()
            # 获得灰度图，并且在内存中创建一个图像对象
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 获取当前帧中的全部人脸
            faces = face_detection.detectMultiScale(gray, 1.3, 5)
            # 对于所有发现的人脸
            for (x, y, w, h) in faces:
                # 在脸周围画一个矩形框，(255,0,0)是颜色，2是线宽
                cv2.rectangle(frame, (x, y), (x + w, y + h), (84, 255, 159), 2)
                # 获取人脸图像
                face = gray[y:y + h, x:x + w]
                '''
                try:
                    # shape变为(48,48)
                    face = cv2.resize(face, (48, 48))
                except:
                    continue
                # 扩充维度，shape变为(1,48,48,1)
                # 将（1，48，48，1）转换成为(1,1,48,48)
                face = np.expand_dims(face, 0)
                face = np.expand_dims(face, 0)
                # 人脸数据归一化，将像素值从0-255映射到0-1之间
                face = preprocess_input(face)
                new_face = torch.from_numpy(face)
                new_new_face = new_face.float().requires_grad_(False)
                # 调用我们训练好的表情识别模型，预测分类
                emotion_arg = np.argmax(emotion_classifier.forward(new_new_face).detach().numpy())
                '''
                emotion_arg=np.argmax(output(face))
                emotion = emotion_labels[emotion_arg]

                emotion_window.append(emotion)

                if len(emotion_window) >= frame_window:  #
                    emotion_window.pop(0)

                try:
                    # 获得出现次数最多的分类
                    emotion_mode = mode(emotion_window)
                except:
                    continue
                emotion_history.append(emotion)
                # 在矩形框上部，输出分类文字
                cv2.putText(frame, emotion_mode, (x, y - 30), font, 1.0, (0, 0, 255), 1, cv2.LINE_AA)
            
            # 将处理后的帧写入新视频文件
            out.write(frame)
            try:
                # 将图片从内存中显示到屏幕上
                cv2.imshow('window_frame', frame)
            except:
                continue
            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        out.release()
        cv2.destroyAllWindows()
        # 使用 moviepy 读取原视频的音频并将其添加到新视频中
        original_clip = VideoFileClip(video_path)
        new_clip = VideoFileClip(output_path).set_audio(original_clip.audio)
        new_clip.write_videofile(output_path.replace('.mp4', '_with_audio.mp4'))

    def immediate():
        emotion_classifier = torch.load("model/model_resnet_final.pkl")
        frame_window = 10

        emotion_window = []
        emotion_history = []

        # 调起摄像头，0是笔记本自带摄像头
        video_capture = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.startWindowThread()
        cv2.namedWindow('immediate face recognition')

        while True:
            # 读取一帧
            _, frame = video_capture.read()
            frame = frame[:,::-1,:]#水平翻转，符合自拍习惯
            frame = frame.copy()
            # 获得灰度图，并且在内存中创建一个图像对象
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 获取当前帧中的全部人脸
            faces = face_detection.detectMultiScale(gray, 1.3, 5)
            # 对于所有发现的人脸
            for (x, y, w, h) in faces:
                # 在脸周围画一个矩形框，(255,0,0)是颜色，2是线宽
                cv2.rectangle(frame, (x, y), (x + w, y + h), (84, 255, 159), 2)

                # 获取人脸图像
                face = gray[y:y + h, x:x + w]

                try:
                    # shape变为(48,48)
                    face = cv2.resize(face, (48, 48))
                except:
                    continue

                # 扩充维度，shape变为(1,48,48,1)
                # 将（1，48，48，1）转换成为(1,1,48,48)
                face = np.expand_dims(face, 0)
                face = np.expand_dims(face, 0)

                # 人脸数据归一化，将像素值从0-255映射到0-1之间
                face = preprocess_input(face)
                new_face = torch.from_numpy(face)
                new_new_face = new_face.float().requires_grad_(False)

                # 调用我们训练好的表情识别模型，预测分类
                emotion_arg = np.argmax(emotion_classifier.forward(new_new_face).detach().numpy())
                emotion = emotion_labels[emotion_arg]

                emotion_window.append(emotion)

                if len(emotion_window) >= frame_window:
                    emotion_window.pop(0)

                try:
                    # 获得出现次数最多的分类
                    emotion_mode = mode(emotion_window)
                except:
                    continue
                emotion_history.append(emotion)
                # 在矩形框上部，输出分类文字
                cv2.putText(frame, emotion_mode, (x, y - 30), font, 1.7, (0, 0, 255), 1, cv2.LINE_AA)

            try:
                # 将图片从内存中显示到屏幕上
                cv2.imshow('immediate face recognition', frame)
            except:
                continue

            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()




    window = tk.Tk()
    window.title("数据科学实践课程设计")
    window.geometry('1200x600')

    # 创建Canvas    
    canvas = tk.Canvas(window, bg="blue", height=1200, width=600)
    canvas.pack()

    # 打开并调整图片大小
    img = Image.open("ins.png")
    img = img.resize((1200, 600), Image.LANCZOS)  # 调整图片大小以适应Canvas
    photo = ImageTk.PhotoImage(img)

    # 设置背景图片
    background_label = tk.Label(window, image=photo)
    background_label.image = photo  # 保持对图片对象的引用
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    # 设置标签
    label_1 = tk.Label(text="数据科学实践课程设计", font=("黑体", 20))
    label_1.place(x=30, y=30)
    label_2 = tk.Label(text="宋子烨", font=("黑体", 15))
    label_2.place(x=950, y=30)
    label_3 = tk.Label(text="基于深度学习的面部表情识别系统", font=("黑体", 35))
    label_3.place(x=250, y=180)

    # 设置按钮
    button1 = tk.Button(text="图片表情识别", command=picture, font=("黑体", 20))
    button2 = tk.Button(text="视频表情识别", command=video, font=("黑体", 20))
    button3 = tk.Button(text="实时表情识别", command=immediate, font=("黑体", 20))

    button1.place(x=170, y=350, width=200)
    button2.place(x=440, y=350, width=200)
    button3.place(x=720, y=350, width=200)

    window.mainloop()

windows()