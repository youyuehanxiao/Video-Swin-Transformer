'''
模型训练
模型：P3D
数据集：UCF-101
'''
import os
import sys
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from swin_transformer_3d_sample import swin_t, swin_s, swin_b, swin_l
from Video_Load import VideoDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as opt
from tqdm import tqdm

#训练类，抽象类，包含基本的训练参数和方法
class Train(ABC):
    def __init__(self, module:nn.Module, optimizer:opt, loss_func=nn.CrossEntropyLoss(),
                 device=torch.device('cuda'), epoches=100, batchsize=2, lr=0.001):
        '''
        :param module: 训练模型
        :param optimizer: 选择的优化器对象
        :param loss_func: 选择的损失函数对象
        :param device: 选择的训练设备
        :param epoches:迭代次数
        '''
        self.module = module #训练模型
        self.optimizer = optimizer #训练优化器
        self.loss_func = loss_func #损失函数
        self.device = device #训练设备
        self.epoches = epoches #迭代次数
        self.batchsize = batchsize #小批量样本数目
        self.lr = lr #学习率

    #保存数据到pth文件
    def save_data_to_pth(self, data: dict, path: str):
        '''
        :param data: 要保存的数据
        :param path: 保存路径，包含文件名
        '''
        torch.save(data, path)

    #保存数据到Excel文件（末尾追加数据）
    def save_data_to_excel(self, data: dict, path: str, write_type='column', sheet_name='Sheet1'):
        '''
        :param data: 要保存的数据
        :param path: 保存路径，包含文件名
        :param write_type: 追加类型，’row'表示按行追加，'column'表示按列追加
        :param sheet: 表示数据保存在哪个工作簿
        '''
        new_data = pd.DataFrame(data) #将字典数据转换成pd.DataFrame类型
        try:
            #以追加方式打开指定excel文件，若文件不存在，新建文件
            with pd.ExcelWriter(path, mode='a', if_sheet_exists='overlay') as wt:
                if write_type == 'row':  # 按行追加
                    try:
                        new_data.to_excel(wt, sheet_name=sheet_name, index=False, header=False,
                                        startrow=wt.book[sheet_name].max_row + 1, startcol=0)  # 在行末尾追加新数据
                    except KeyError: #指定sheet_name不存在
                        new_data.to_excel(wt, sheet_name=sheet_name, index=False, header=True)  # 新建sheet_name，并写入数据
                elif write_type == 'column':  # 按列追加
                    try:
                        new_data.to_excel(wt, sheet_name=sheet_name, index=False, header=True,
                                          startrow=0, startcol=wt.book[sheet_name].max_column + 1)  # 在行末尾追加新数据
                    except KeyError:
                        new_data.to_excel(wt, sheet_name=sheet_name, index=False, header=True)  # 新建sheet_name，并写入数据
                else:
                    print('未被实现的追加方式！')
                    sys.exit(-1)
        except FileNotFoundError or FileExistsError: #指定文件不存在，创建文件并保存数据
            with pd.ExcelWriter(path, mode='w') as wt:
                new_data.to_excel(wt, sheet_name=sheet_name, index=False, header=True) #第一次添加新数据，写入头部属性名

    #保存数据到文件
    def save_data_to_file(self, data: dict, save_path: str, filename: str, **kwargs):
        '''
        :param data: 要保存的数据
        :param save_path: 要保存路径
        :param filename: 要保存的文件名（目前仅支持pth和xlsx格式保存）
        :param **kwargs: 当保存为excel文件时的一些参数
        '''
        # 判断路径是否存在，若不存在，创建路径
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 获取文件扩展名
        ext = os.path.splitext(filename)[1]
        # 判断文件类型，保存数据
        if ext == '.pth':  # 保存为pth文件，直接进行保存
            self.save_data_to_pth(data, os.path.join(save_path, filename))
        elif ext == '.xlsx':  # 保存到excel表格，在末尾进行追加
            self.save_data_to_excel(data, os.path.join(save_path, filename), **kwargs)
        else:
            print('未被实现的保存格式！')
            sys.exit(-1)

    #数据集加载
    @abstractmethod
    def data_load(self):
        pass

    #训练
    @abstractmethod
    def train(self):
        pass

#UCF-101视频数据，P3D模型训练
class UCF101_train(Train):
    def __init__(self, video_root:str, split_root:str, clip_crop_size=(16, 160, 160), resize=(182, 242),
                                  num_mode=1, video_type='RGB', clip_type_param=('rand', 2), **kwargs):
        super(UCF101_train, self).__init__(**kwargs)
        self.video_root = video_root #视频数据根目录
        self.split_root = split_root #数据划分文件根目录
        self.clip_crop_size = clip_crop_size #剪辑后大小
        self.resize = resize #帧图像缩放后的大小
        self.num_mode = num_mode #使用的数据块编号
        self.video_type = video_type #视频类型
        self.clip_type_param = clip_type_param #视频剪辑方式及参数

        #加载数据
        self.train_data, self.train_loader, \
            self.test_data, self.test_loader = self.data_load()
        # print(self.train_data.list_label_id)
        # print(self.test_data.list_label_id)
        #创建测试结果字典
        self.dict_test_result = self.create_dict_test_result()
        #训练过程
        #self.train()

    #创建测试结果字典
    def create_dict_test_result(self):
        '''
        :return dict_test_result: 测试结果字典，长度为数据类别数目，每个元素表示为：label:[pre_num, true_pre_num]
        '''
        dict_test_result = {}
        for id in range(self.test_data.num_class):
            dict_test_result[id] = [0, 0]

        return dict_test_result

    #更新测试结果字典
    def update_test_result(self, pred_class:torch.Tensor, labels:torch.Tensor):
        '''
        :param pre_class: 一个bach预测类别
        :param labels: 真实类别
        '''
        #判断是否预测正确，若正确为True，否则为False
        answer = list(torch.eq(pred_class, labels))
        #更新测试结果字典
        for index, label_id in enumerate(list(pred_class)):
            self.dict_test_result[label_id.item()][0] += 1 #预测为id类的数目加1
            if answer[index]:
                self.dict_test_result[label_id.item()][1] += 1 #正确预测为id类的数目加1

    #重置测试结果字典
    def reset_test_result(self):
        #将每个类别的预测数目和预测正确的数目置为0
        for key in self.dict_test_result.keys():
            self.dict_test_result[key][0] = 0
            self.dict_test_result[key][1] = 0

    #数据加载
    def data_load(self):
        #加载训练集
        train_data = VideoDataset(self.video_root, self.split_root, clip_crop_size=self.clip_crop_size,
                                       resize=self.resize, mode='train', num_mode=self.num_mode,
                                       video_type=self.video_type, clip_type_param=self.clip_type_param)
        train_loader = DataLoader(train_data, batch_size=self.batchsize, shuffle=True)

        #加载测试集
        test_data = VideoDataset(self.video_root, self.split_root, clip_crop_size=self.clip_crop_size,
                                      resize=self.resize, mode='test', num_mode=self.num_mode,
                                      video_type=self.video_type, clip_type_param=self.clip_type_param)
        test_loader = DataLoader(test_data, batch_size=self.batchsize, shuffle=False)

        return train_data, train_loader, test_data, test_loader

    #模型训练
    def train(self):
        #将模型加入设备
        self.module.to(self.device)

        max_acc = 0  #记录最大准确率
        best_epoch = None #记录产生最大准确率时的epoch
        best_params = None #记录最优参数

        #训练开始
        print(f'使用{self.train_data.__len__()}个视频进行训练，{self.test_data.__len__()}个视频进行测试')
        print('开始训练......')
        for epoch in range(self.epoches):
            print(f'epoch[{epoch + 1}/{self.epoches}]:')
            #训练过程
            self.module.train() #训练模式
            train_bar = tqdm(self.train_loader, file=sys.stdout, ncols=80) #创建一个进度条
            train_acc_num = 0 #记录训练时预测正确的数量
            #遍历训练集，批量训练
            for videos, labels in train_bar:
                out = self.module(videos.to(self.device)) #正向计算
                pre_class = torch.max(out, dim=1)[1] #模型预测的类别
                train_acc_num += torch.eq(pre_class, labels.to(self.device)).sum().item() #与真实类别比较，若相同为True，不同为False；计算出预测正确的数量
                loss = self.loss_func(out, labels.to(self.device)) #计算损失
                self.optimizer.zero_grad() #梯度清零
                loss.backward() #反向传播，自动求导
                self.optimizer.step() #梯度下降，更新参数
                #输出进度条描述信息
                # print('labels:', labels)
                # print('loss:', loss)
                # print(loss.item())
                train_bar.desc = f'train loss:{loss.item()}'
            #计算训练集准确
            train_acc = round(train_acc_num / self.train_data.__len__(), 3)
            #测试过程
            self.module.eval() #测试模式
            test_acc_num = 0 #记录测试过程预测正确的数量
            self.reset_test_result() #重置测试结果字典
            #开始测试
            test_bar = tqdm(self.test_loader, file=sys.stdout, ncols=80) #创建测试进度条
            with torch.no_grad():
                for videos, labels in test_bar:
                    out = self.module(videos.to(self.device)) #预测
                    pre_class = torch.max(out, dim=1)[1] #预测类别
                    self.update_test_result(pred_class=pre_class.cpu(), labels=labels) #更新测试结果字典
                    test_acc_num += torch.eq(pre_class, labels.to(self.device)).sum().item() #预测正确的数目
                    test_bar.desc = f'test acc num:{test_acc_num}'
            #计算当前epoch测试结束后，每个类别的预测查准率precision、召回率recall和准确率accuration
            test_acc = round(test_acc_num / self.test_data.__len__(), 3) #测试集准确率
            valuation_norm = {
                f'precision_{epoch + 1}': [round(p_t / p_n, 3) if p_n != 0 else 0.0 for p_n, p_t in self.dict_test_result.values()],
                f'recall_{epoch + 1}': [round(self.dict_test_result[key][1] / self.test_data.dict_num_everylabel[key], 3)
                                        if self.test_data.dict_num_everylabel[key] != 0 else 0.0
                                        for key in self.dict_test_result.keys()],
                f'accuration_{epoch + 1}': [test_acc for i in range(self.test_data.num_class)]}
                #f'accuration_{epoch + 1}': test_acc}
            # for v in valuation_norm.values():
            #     print(len(v), v)
            #print(valuation_norm)
            #将评估指标valuation_norm存入excel文件
            if epoch == 0: #如果是第一次写入，创建新文件并将标签名称写入第一列
                self.save_data_to_file({'label_name': self.test_data.dict_class_name_id.keys()}, './result', 'valuation_norm.xlsx')
            self.save_data_to_file(valuation_norm, './result', 'valuation_norm.xlsx',
                                   write_type='column', sheet_name='Sheet1')
            #更新最大准确率和最优参数
            if test_acc > max_acc:
                max_acc = test_acc
                best_epoch = epoch + 1
                #best_params = self.module.state_dict() #错误编写，必须要deepcopy
                #存储当前最优权重
                self.save_data_to_file({'weight': self.module.state_dict()}, './result', f'weight_{best_epoch}.pth')

            #输出训练精度、测试精度和最大测试精度
            print(f'train_acc:{train_acc}  test_acc:{test_acc}  max_acc:{max_acc}')
        #存储最优参数到pth文件
        #self.save_data_to_file({'weight': best_params}, './result', f'weight_{best_epoch}.pth')

        print('训练结束！')
        print(f'best_epoch：{best_epoch}  max_acc：{max_acc}')

    #推理
    def infer(self, weight):
        #加载权重
        self.module.load_state_dict(weight, strict=True)
        self.module.to(self.device)

        #推理
        self.module.eval() #推理测试模式
        infer_acc_num = 0 #记录推理测试过程预测正确的数量
        #开始推理测试
        infer_bar = tqdm(self.test_loader, file=sys.stdout, ncols=80) #创建推理测试进度条
        with torch.no_grad():
            for video, labels in infer_bar:
                out = self.module(video.to(self.device)) #预测
                pre_class = torch.max(out, dim=1)[1] #预测类别
                infer_acc_num += torch.eq(pre_class, labels.to(self.device)).sum().item() #预测正确的数目
                infer_bar.desc = f'infer acc num:{infer_acc_num}'
        #计算准确率
        infer_acc = round(infer_acc_num / self.test_data.__len__(), 3)
        print(f'infer_acc:{infer_acc}')

def main():
    video_root = 'D:/Machine learning/视频特征提取/datasets/UCF-101/Videos' #视频数据根目录
    split_root = 'D:/Machine learning/视频特征提取/datasets/UCF-101/Train_Test_list' #训练集测试集划分文件根目录

    # #加载训练集
    # train_data = VideoDataset(video_root, split_root, clip_crop_size=(16, 160, 160), resize=(182, 242),
    #                           mode='train', num_mode=(1, 2, 3), video_type='RGB', clip_type_param=('rand', 2))
    # train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    #
    # #加载测试集
    # test_data = VideoDataset(video_root, split_root, clip_crop_size=(16, 160, 160), resize=(182, 242),
    #                           mode='test', num_mode=(1, 2, 3), video_type='RGB', clip_type_param=('rand', 2))
    # test_loader = DataLoader(test_data, batch_size=2, shuffle=False)

    #定义训练设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'use device：{device}')

    #定义模型并加入设备
    #module = P3D199(pretrained=False, modality='RGB', num_classes=101).to(device)
    module = swin_t(num_class=101).to(device)

    #定义损失函数
    loss_func = nn.CrossEntropyLoss() #交叉熵损失

    #定义优化器
    optimizer = opt.SGD(params=module.parameters(), lr=0.001, momentum=0.9) #随机梯度下降

    # #定义训练结果存储目录
    # result_savepath = './result'
    # #如果目录不存在，自动创建目录
    # if not os.path.exists(result_savepath):
    #     os.mkdir(result_savepath)

    #训练过程
    train = UCF101_train(video_root, split_root, clip_type_param=('rand', 3), module=module, optimizer=optimizer,
                         loss_func=loss_func, batchsize=2, device=device)
    train.train()
    #weight_dict = torch.load('./result_noa/weight_85.pth').get('weight')
    #train.infer(weight_dict)

    #测试保存文件
    # data = {'age': [1, 2, 3]}
    # data2 = {'age': [4, 5, 6]}
    # train.save_data_to_file(data=data2, save_path='./', filename='bbb.xlsx', sheet_name='Sheet2')

if __name__ == '__main__':
    main()




