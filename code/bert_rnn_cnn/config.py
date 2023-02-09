import os.path
import torch
import time
'''换网络运行只需要更换self.mynet=这个参数即可，其他根据情况微调'''

class Config(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.3     # 随机失活
        self.require_improvement = 2000  # 若超过2000batch效果还没提升，则提前结束训练
        self.num_classes0 =2  # 第一个标签类别数，无需修改
        self.num_classes1= 3# 第二个标签类别数，无需修改
        self.num_epochs = 20   # epoch数
        self.pad_size = 400 # 每句话处理成的长度(短填长切) 根据平均长度来
        self.bert_learning_rate = 1e-5   # bert的学习率，minirbt-h256需要用更大的学习率例如1e-4,其他bert模型设置为1e-5较好
        self.other_learning_rate = 2e-5#其他层的学习率
        self.frac=1#使用数据的比例，因为训练时间长，方便调参使用,1为全部数据，0.1代表十分之一的数据
        self.embed =128
        self.n_vocab=21128#词表大小，因为都采用bert词表，这里是固定的
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256

        self.bert_name='chinese-bert-wwm-ext' #bert类型
        self.bert_fc=768

        self.mynet='bert'#网络类型 bert rnn cnn三种
        if self.mynet=='bert':
            self.batch_size = 8  # mini-batch大小，看显存决定
        else:
            self.batch_size = 16  # mini-batch大小，看显存决定
        if not os.path.exists('model'):
            os.makedirs('model')

        self.save_path = 'model/'+self.mynet+'.pt'##保存模型的路径
        self.log_dir= './log/'+self.mynet+'/'+str(time.time())#tensorboard日志的路径


