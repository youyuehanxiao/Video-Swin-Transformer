'''
一、UCF-101视频数据存放目录：
1. 数据根目录：D:/Machine learning/视频特征提取/datasets/UCF-101
2. 视频资源目录：UCF-101/Videos
    2.1. Videos目录下有101个子目录，子目录下分别存放101个类别的视频，每个子目录名称为视频类别名
3. 训练集-测试集划分文件的目录：UCF-101/Train_Test_list
    3.1. Train_Test_list目录下有7个文件：3个训练集文件（存放训练集视频路径和类别编号）、3个测试集文件（存放测试集视频路径，没有类别编号）、1个类别标签文件（存放标签名-编号映射）

二、UCF-101数据加载：
1. 创建（类别-编号）映射字典dict_class_name_id
2. 创建训练集文件列表list_train_filename（存储训练集文件路径）、对应标签名列表list_train_classname和对应标签编号列表list_train_classid
3. 创建测试集文件列表list_test_filename（存储测试集文件路径）、对应标签名列表list_test_classname和对应标签编号列表list_test_classid
4. 根据给定的索引加载对应视频（获取视频帧，并进行处理）

三、P3D原文的数据处理
1. ResNet50
    1.1 视频帧大小重置为：240×320
    1.2 随机裁剪为：224×224
    1.3 冻结了除第一个之外的所有 Batch Normalization 层的参数（处第一个外，其与参数不进行学习）
    1.4 添加了一个额外的 dropout 层，其 dropout 率为 0.9，以减少过拟合的影响
2. P3D ResNet
    2.1 视频尺寸重置为：16 × 182 × 242
    2.2 随机裁剪为：16 × 160 × 160
    2.3 每个帧/剪辑都是随机的水平翻转以进行增强
    2.4 每个小批量设置为128帧/剪辑，这是用多个GPU并行实现的。网络参数采用标准SGD优化，初始学习率设置为0.001，每3K次迭代后除以10。 7.5K 次迭代后训练停止。

四、视频剪辑（剪辑长度为clip_len）策略
1. 剪辑策略需要根据数据集的情况进行设计——剪辑出重要的片段
2. 或许可以用深度学习的方式自动学习出重要的片段
3. 本次采用的策略是分段剪辑和随机剪辑
    3.1 分段剪辑
    a. 定义分的段数目num_sec
    b. 计算每一段选取的帧数si：clip_len // num_sec，最后一段为clip_len // num_sec+clip_len % num_sec
    c. 计算源数据每一段的帧数（总帧数为count）di：count // num_sec，最后一段为count // num_sec+count % num_sec
    d. 计算各段的选取步长并进行选取：步长——di // si；从当前段首帧开始选取（包含首帧）
    3.2 随机剪辑
    a. 设置选取步长step，并随机选取一个帧编号作为开始帧start（要保证后面的帧数足够进行剪辑）
    b. 根据剪辑长度len_clip，计算出剪辑范围的上限帧end（不含）
    c. 在[start,end)范围内，从左到右，以步长为step依次选取帧

*有些视频会出现某些帧不能用的情况，所以视频总帧数不一定等于有用帧数
'''

import os
import sys
import time

import random
import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class VideoDataset(Dataset):
    def __init__(self, video_root:str, split_root:str, clip_crop_size:tuple, resize:tuple,
                 mode:str, num_mode:int|tuple|list, video_type='RGB', clip_type_param=('rand', 2)):
        '''
        :param video_root: 视频数据根目录
        :param split_root: 划分数据的文件根目录
        :param clip_crop_size: 视频剪辑后的大小（帧数，每一帧图像的高，每一帧图像的宽）
        :param resize: 对单帧图像缩放后的大小（图像高，图像宽）
        :param mode: 加载的数据模块种类（训练集train或测试集test）
        :param num_mode: 加载哪个模块（train01,train02,train03 or test01,test02,test03）
        :param video_type: 视频类型（彩色RGB或灰度GRAY）
        :param clip_type_param: 剪辑类型及参数('sec', 分段数目) 或 ('rand', 步长)
        '''

        self.clip_crop_size = clip_crop_size #剪辑或裁剪后的视频尺寸
        self.resize = resize #对单帧图像缩放后的尺寸
        self.mode = mode #加载的模块种类（train or test）
        self.num_mode = num_mode #加载哪个模块
        self.video_type = video_type #视频类型
        self.clip = clip_type_param #剪辑类型及参数

        #创建类（别名->数字编号）的映射字典、视频文件名（路径）列表、视频对应标签名列表
        self.dict_class_name_id, self.list_video_filename, self.list_label_name = self.create_list_dataproperty(video_root, split_root)
        #self.list_video_filename = self.list_video_filename[0:100]
        #self.list_label_name = self.list_label_name[0:100]
        #加载视频对应标签id（数字编号）列表
        self.list_label_id = self.create_list_label_id()
        #创建每个类别对应样本数目的字典
        self.dict_num_everylabel = self.countnum_ererylabel(self.num_class)

    #根据索引，加载视频数据和对应标签
    def __getitem__(self, index):
        '''
        :param index: 数据索引
        :return video, labelid: 视频数据和类别id
        '''
        video = self.load_video(self.list_video_filename[index]) #加载并处理指定视频数据
        labelid = self.list_label_id[index]
        return video, labelid

    # 获取数据总量
    def __len__(self):
        return len(self.list_video_filename)

    #创建类别（名称-编号）映射字典、文件名（路径）列表、对应类别名称列表
    def create_list_dataproperty(self, root:str, path:str) -> tuple[dict,list,list]:
        '''
        :param root: 视频源文件存放的根目录
        :param path: 划分数据的文件根目录
        '''
        filenames = os.listdir(path) #获取path路径下的文件名
        class_name_id, video_filename, label_name = {}, [], [] #种类名-编号映射；视频路径；视频对应类别标签名

        #创建类别名-标签映射
        for file in filenames:
            if file.find('class') >= 0: #classname -> id
                #读取txt文件，创建类别名-标签映射
                with open(os.path.join(path, file)) as fp:
                    line = fp.readline() #读取第一行
                    while line: #循环行读取
                        l_line = line.strip().split(' ') #清除首尾空白，并以空格分割成列表（列表中包含两个字符串，第一个为标签名，第二个为编号——从1开始）
                        class_name_id[l_line[1]] = int(l_line[0]) - 1 #创建标签名-数字编号对应字典
                        line = fp.readline() #读取下一行
                    fp.close() #读取完毕，关闭文件

                filenames.remove(file) #处理完成，删除文件名
                break


        #搜索待处理文件
        processing_filenames = []  #待处理文件列表
        if isinstance(self.num_mode, int):  #仅仅加载一个数据模块
            for file in filenames:  #找到指定的模块
                if file.find(self.mode) >= 0 and file.find(str(self.num_mode)) >= 0:
                    processing_filenames.append(file) #将文件名添加到待处理列表
                    break
        else: #加载多个数据模块
            for x in self.num_mode:
                for file in filenames:
                    if file.find(self.mode) >= 0 and file.find(str(x)) >= 0:  #找到指定文件名
                        processing_filenames.append(file)  #将文件名添加到待处理列表
                        filenames.remove(file) #文件名已添加到待处理列表，删除原列表中的该文件名
                        break

        #创建视频文件路径列表和对应标签名列表
        if len(processing_filenames) == 0:
            print('指定文件不存在！')
            sys.exit(-1)

        for file in processing_filenames:
            with open(os.path.join(path, file), 'r') as fp: #打开文件
                line = fp.readline() #读取第一行
                while line: #循环读取，直到文件末尾
                    line = line.strip().split(' ')[0] #清除首尾空白，将读取到的一行按空格分割成列表（第一个字符串为文件相对路径），获取文件相对路径
                    i = line.find('/') #找到'/'的位置，'/'之前的字符串是视频对应标签名
                    label_name.append(line[0:i]) #将标签添加到列表
                    video_filename.append(os.path.join(root, line)) #将视频绝对路径添加到列表
                    line = fp.readline() #读取下一行
                fp.close() #读取完毕，关闭文件

        return class_name_id, video_filename, label_name

    #根据视频标签名列表创建对应标签编号列表
    def create_list_label_id(self, dict_name_id_map=None, list_label_name=None) -> list:
        '''
        :param name_id_map: 名称-编号映射字典
        :param label_name: 视频标签名列表
        :return list_label_id: 标签名对应的标签编号列表
        '''
        list_label_id = [] #存储生成的标签编号
        if list_label_name: #使用传入的标签名列表
            if dict_name_id_map: #使用传入的标签名-编号映射列表
                for label in list_label_name:
                    list_label_id.append(dict_name_id_map[label]) #得到当前标签名对应编号
            else: #不使用传入的签名-编号映射列表，使用对象内的签名-编号映射
                for label in list_label_name:
                    list_label_id.append(self.dict_class_name_id[label])
        else: #不使用传入的标签名列表，使用对象内的标签名列表
            if dict_name_id_map: #使用传入的标签名-编号映射列表
                for label in self.list_label_name:
                    list_label_id.append(dict_name_id_map[label]) #得到当前标签名对应编号
            else: #不使用传入的签名-编号映射列表，使用对象内的签名-编号映射
                for label in self.list_label_name:
                    list_label_id.append(self.dict_class_name_id[label])

        return list_label_id

    #加载并处理指定视频数据
    def load_video(self, video_path:str):
        '''
        :param video_path: 将要加载和处理的视频数据
        :return video: 处理后的视频数据（四维张量）
        '''
        #定义对每一帧图像进行的处理操作
        transform = transforms.Compose([
            transforms.Resize((self.resize[0], self.resize[1])), #图像缩放
            transforms.RandomCrop((self.clip_crop_size[1], self.clip_crop_size[2])) if self.mode == 'train' else
            transforms.Resize((self.clip_crop_size[1], self.clip_crop_size[2])), #训练集随机裁剪，测试集无操作
            transforms.ToTensor(), #转换成0.0-1.0的张量
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) if self.video_type == 'RGB' else
            transforms.Normalize(0.5, 0.5) #标准化
        ])

        #指定一个待处理的视频
        v_path = video_path #要加载的视频路径
        path_selected = [] #已经选择的无效视频
        video_stream, frame_count, frames = None, None, None #要处理的视频流对象、可用帧数目、存储可用帧的列表
        #选择一个有效视频
        while True:
            try:
                video_stream = cv2.VideoCapture(v_path) #创建视频流对象
                frame_count, frames = self.frames_read(video_stream) #读取可用帧
                # frame_count = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT)) #获取视频帧数
                if (self.clip[0] == 'sec' and frame_count >= self.clip_crop_size[0]) or \
                        (self.clip[0] == 'rand' and frame_count >= self.clip_crop_size[0] * self.clip[1]):
                    break #视频加载成功，跳出循环
                elif self.clip[0] != 'sec' and self.clip[0] != 'rand':
                    print('不被支持的剪辑类型！')
                    sys.exit(-1)
                else: #当前视频帧数过少，重新选择
                    path_selected.append(v_path)  #将无效视频标记为已选择，下次不再选择它
                    v_path = self.list_video_filename[np.random.randint(0, self.__len__())]  #随机获取一个视频路径
                    while v_path in path_selected:  #选择的视频无效，重新选择
                        v_path = self.list_video_filename[np.random.randint(0, self.__len__())] #随机获取一个视频路径
            except RuntimeError: #当前视频加载出错，尝试加载其它视频
                path_selected.append(v_path) #将无效视频标记为已选择，下次不再选择它
                v_path = self.list_video_filename[np.random.randint(0, self.__len__())] #随机获取一个视频路径
                while v_path in path_selected: #选择的视频无效，重新选择
                    v_path = self.list_video_filename[np.random.randint(0, self.__len__())] #随机获取一个视频路径

        #判断视频流是否可用
        if not video_stream:
            print('无效视频！')
            sys.exit(-1)
        video_stream.release() #释放视频流对象空间

        #视频剪辑和帧处理
        if self.clip[0] == 'sec': #分段剪辑
            list_video = self.clip_section(frames, frame_count, num_clip=self.clip_crop_size[0],
                                           num_section=self.clip[1], transform=transform)
        elif self.clip[0] == 'rand': #随机剪辑
            list_video = self.clip_random(frames, frame_count, num_clip=self.clip_crop_size[0],
                                          step_choice=self.clip[1], transform=transform)
        else:
            print('不被支持的剪辑类型：', self.clip[0])
            sys.exit(-1)

        #print(len(list_video))
        #尾部处理
        video = torch.stack(list_video) #转换为张量
        #维度调换（t × c × h × w -> c × t × h × w）
        return video.transpose(0, 1)

    #读取视频可用帧
    def frames_read(self, video_stream:cv2.VideoCapture):
        '''
        :param video_stream: 视频流对象
        :return num_frame, frames: 可用帧数目和可用帧列表
        '''
        num_frame = 0 #可用帧数目
        frames = [] #存储可用帧列表
        statu, frame = video_stream.read() #读取第一帧
        while statu:
            frames.append(frame)
            num_frame += 1
            statu, frame = video_stream.read() #读取下一帧

        return num_frame, frames

    #分段剪辑
    def clip_section(self, frames:list, frame_count:int, num_clip:int, num_section:int, transform:transforms):
        '''
        :param frames: 可用视频帧集合
        :param frame_count: 可用帧总数目
        :param num_clip: 剪辑后的帧数目
        :param num_section: 分段数目
        :param transform: 帧处理操作集合
        :return list_video: 剪辑后的视频帧列表
        '''
        #判断帧数与实际是否相同
        if frame_count != len(frames):
            print('视频帧数与实际不符！')
            sys.exit(-1)

        #分段剪辑并处理每一帧图像
        list_video = [] #存储选择的每一帧图像（图像已进行处理）
        #分段剪辑参数
        #print('总帧数：', frame_count)
        l_fcount = frame_count // num_section  #除最后一段之外，每一段的帧数
        r_fcount = l_fcount + frame_count % num_section  # 最后一段的帧数
        l_choicenum = num_clip // num_section  #除最后一段之外，每一段需要选择的帧数
        r_choicenum = l_choicenum + num_clip % num_section  #最后一段需要选择的帧数
        l_step = l_fcount // l_choicenum  #除最后一段之外，每一段的步长
        r_step = r_fcount // r_choicenum  #最后一段的步长
        current_count, choiced_count = 0, 0  #当前读取的帧id、已选择的帧数目
        #依次遍历所有帧，分段选取并处理
        for current_count, frame in enumerate(frames):
            if choiced_count == num_clip: #剪辑完成，跳出循环
                break
            #判断当前帧是否选取，若选取，则进行处理
            id_section = current_count // l_fcount  # 当前帧在第几段
            if id_section < num_section - 1:  #除最后一段之外
                #print(f'当前第{id_section}段，', end='')
                if choiced_count - l_choicenum * id_section < l_choicenum and \
                        (current_count - l_fcount * id_section) % l_step == 0:  #当前段已选择的帧数小于需要选择的帧数，并且当前帧与已选择的前一帧的间隔满足条件，选择并进行处理
                    frame_p = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB if self.video_type == 'RGB' else None))) #处理当前帧
                    list_video.append(frame_p)  #将处理好的帧添加到选择列表
                    choiced_count += 1 #已选择帧数加1
                #print(f'已选择{choiced_count - l_choicenum * id_section}帧，需要{l_choicenum}帧')
            else: #最后一段
                #print(f'当前第{current_count}帧，第{id_section}段，步长{r_step}；', end='')
                if choiced_count - l_choicenum * (num_section - 1) < r_choicenum and \
                        (current_count - l_fcount * (num_section - 1)) % r_step == 0: #当前段已选择的帧数小于需要选择的帧数，并且当前帧与已选择的前一帧的间隔满足条件，选择并进行处理
                    frame_p = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB if self.video_type == 'RGB' else None))) #处理当前帧
                    list_video.append(frame_p)  #将处理好的帧添加到选择列表
                    choiced_count += 1  #已选择帧数加1
                #print(f'已选择{choiced_count - l_choicenum * (num_section - 1)}帧，需要{r_choicenum}帧')

        return list_video

    #随机剪辑
    def clip_random(self, frames:list, frame_count:int, num_clip:int, step_choice:int, transform:transforms):
        '''
        :param frames: 可用视频帧集合
        :param frame_count: 可用帧总数目
        :param num_clip: 剪辑后的帧数目
        :param step_choice: 选取步长
        :param transform: 帧处理操作集合
        :return list_video: 剪辑后的视频帧列表
        '''
        #判断帧数与实际是否相同
        if frame_count != len(frames):
            print('视频帧数与实际不符！')
            sys.exit(-1)

        #随机剪辑并处理帧
        list_video = [] #剪辑并处理后的视频帧
        start_id = random.randint(0, frame_count - num_clip * step_choice) #随机一个开始帧id（保证后面的帧数足够进行剪辑）
        end_id = start_id + num_clip * step_choice #结束帧id的最大值（不含）
        #从start_id开始，按照步长step_choice从前往后依次进行帧选取和处理
        for i in range(start_id, end_id, step_choice):
            #处理当前帧
            frame_p = transform(Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB if self.video_type == 'RGB'
            else cv2.COLOR_BGR2GRAY)))
            #将当前帧添加到剪辑列表
            list_video.append(frame_p)

        return list_video

    #统计每个类别的样本数目
    def countnum_ererylabel(self, num_class: int):
        '''
        :param num_class: 类别数目
        :return dict_num:everylabel: 每个类别id对应样本数目的字典，长度为num_class
        '''
        # 初始化字典
        dict_num_everylabel = {}
        for id in range(num_class):
            dict_num_everylabel[id] = 0

        # 统计每个类别的数目
        for label in self.list_label_id:
            dict_num_everylabel[label] += 1

        return dict_num_everylabel

        # 获取类别总数

    #获取类别数目
    @property
    def num_class(self):
        return len(self.dict_class_name_id)



if __name__ == '__main__':
    video_root = 'D:/Machine learning/视频特征提取/datasets/UCF-101/Videos'
    split_root = 'D:/Machine learning/视频特征提取/datasets/UCF-101/Train_Test_list'
    mydata = VideoDataset(video_root=video_root, split_root=split_root, clip_crop_size=(16, 160, 160), resize=(182, 242),
                          mode='test', num_mode=1, video_type='RGB', clip_type_param=('rand', 2))
    mydata_loader = DataLoader(mydata, batch_size=100, shuffle=False)

    # print(f'数据总量：{mydata.__len__()}；类别数目：{mydata.num_class}')
    # print(f'每个类别的数目：{mydata.dict_num_everylabel}')
    # print(sum(mydata.dict_num_everylabel.values()))
    #print(mydata.__len__())
    time1 = time.time()
    for batch, (data, label) in enumerate(mydata_loader):
        print(f'第{batch}批：{data.size(), label}')
    time2 = time.time()
    print(f'加载数据用时{time2 - time1}s')
    #print(1543.12619638443 / 60)























