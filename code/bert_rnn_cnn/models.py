import torch.nn as nn
from transformers import BertModel
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.utils.model_zoo as model_zoo



class Mybert(nn.Module):
    def __init__(self,config):
        super(Mybert, self).__init__()
        self.config=config
        self.bert= BertModel.from_pretrained(self.config.bert_name)#bert的种类
        self.fc_0 = nn.Linear(self.config.bert_fc, self.config.num_classes0)
        self.fc_1 = nn.Linear(self.config.bert_fc, self.config.num_classes1)
        self.drop=nn.Dropout(self.config.dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,inx):
        # BERT
        tokens=inx
        outputs = self.bert(tokens)
        #emb (32,128)-(32,768)
        pooled_output = outputs[1]
        pooled_output=self.drop(pooled_output)

        logits0 = self.fc_0(pooled_output)
        logits0=self.softmax(logits0)

        logits1 = self.fc_1(pooled_output)
        logits1=self.softmax(logits1)

        return logits0,logits1


class Mycnn(nn.Module):
    def __init__(self, config):
        super(Mycnn, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed)  ###生成词向量表格
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])## K 表示卷积核的宽 embed为长
        self.dropout = nn.Dropout(config.dropout)
        self.fc_0 = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes0)
        self.fc_1 = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes1)
        self.softmax = nn.Softmax(dim=1)
        ## nn.Sequential()

    def conv_and_pool(self, x, conv):  #卷积后大小=数据维度-减去卷积维度+1
        x=conv(x)## bs*1也就是输入图层*seq_len*embedding_dim  example: [128,1,32,300] [128,300,32]  [32,300] -[2,300] -[31,1]
        x = F.relu(x) ## bs*输出图层大小    最后一个维度会变成1，因为卷积核的大小是k*embedding_dim    example:[128, 256, 31, 1]
        x = x.squeeze(3)## 把最后一个维度的1去掉  example:[128, 256, 31]
        x = F.max_pool1d(x, x.size(2))##example:[128, 256, 1]
        x = x.squeeze(2)  #[128,256]
        return x

    def forward(self, x):
        #128 32————128*1*32*300
        out = self.embedding(x) ## 进来的x[0]就是外面的trains:shape is bs*seq_len   out的输出为bs*seq_len*embedding_dim
        out = out.unsqueeze(1)## 在1的位置加入一个1，这是因为在textccnn中，我们的输入类别到图像里面是一个灰度图，也就是1个层面的，不是RGB那种三个层面的；
        # 所以上面这个out输出之后的维度是bs*1*seq_len*embedding_dim
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)##池化后的级联
        out = self.dropout(out) ##dropout函数防止过拟合18*8


        logits0 = self.fc_0(out)
        logits0 = self.softmax(logits0)

        logits1 = self.fc_1(out)
        logits1 = self.softmax(logits1)
        return logits0,logits1


class Myrnn(nn.Module):
    def __init__(self, config):
        super(Myrnn, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed)  ###生成词向量表格
        self.rnn=nn.LSTM(config.embed, # x的特征维度,即embedding_dim
                            128,
                            2, # 把lstm作为一个整体，然后堆叠的个数的含义
                            batch_first=True,
                            bidirectional=False
                            )
        self.dropout = nn.Dropout(config.dropout)
        self.fc_0 = nn.Linear(128, config.num_classes0)
        self.fc_1 = nn.Linear(128, config.num_classes1)
        self.softmax = nn.Softmax(dim=1)
        ## nn.Sequential()


    def forward(self, x):
        #128 32————128*1*32*300
        out = self.embedding(x) ## 进来的x[0]就是外面的trains:shape is bs*seq_len   out的输出为bs*seq_len*embedding_dim
        out,(_,_)=self.rnn(out)
        out = self.dropout(out) ##dropout函数防止过拟合
        #print(out[:,-1,:].shape)
        # out = self.fc(out[:,-1,:])
        # logits = self.softmax(out)
        logits0 = self.fc_0(out[:,-1,:])
        logits0 = self.softmax(logits0)

        logits1 = self.fc_1(out[:,-1,:])
        logits1 = self.softmax(logits1)
        return logits0,logits1