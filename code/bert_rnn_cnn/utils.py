import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
import cv2
from transformers import BertTokenizer
from config import Config
# a.通过词典导入分词器
#"bert-base-chinese"

#bert_model/chinese-bert-wwm-ext

#tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
class My_Dataset(Dataset):
    def __init__(self,path,config,iftrain):#### 读取数据集
        self.config=config
        #启用训练模式，加载数据和标签
        self.iftrain=iftrain
        self.df = pd.read_csv(path)
        self.text = self.df['微博正文'].to_list()
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_name)

        #启用训练模式，加载数据和标签
        if self.iftrain==1:
            self.labels0=self.df['label0'].to_list()
            self.labels1 = self.df['label1'].to_list()

    def __getitem__(self, idx):
        text=self.text[idx]
        try:
            len(text)#部分文本是nan
        except:
            text=''


        text=self.tokenizer(text=text, add_special_tokens=True,
                  max_length=self.config.pad_size,  # 最大句子长度
                  padding='max_length',  # 补零到最大长度
                  truncation=True)
        #print(text)
        # 中文-英文  （t1[我 吃 饭],t2[i eat food]）  [[0,0,0,0,0],[1,1,1,1,1]]
        #text 三个部分  token_type_ids(句子对 中文句子 英文句子)
        input_id= torch.tensor(text['input_ids'], dtype=torch.long)
        #attention_mask = torch.tensor(text['attention_mask'], dtype=torch.long)#可用可不用
        #

        if self.iftrain==1:
            label0=int(self.labels0[idx])
            label0 = torch.tensor(label0, dtype=torch.long)

            label1=int(self.labels1[idx])
            label1 = torch.tensor(label1, dtype=torch.long)
            return input_id.to(self.config.device),(label0.to(self.config.device),label1.to(self.config.device))

        elif self.iftrain==0:
            return input_id.to(self.config.device)

    def __len__(self):
        return len(self.df)#总数据长度

def get_time_dif(start_time):

    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

if __name__=='__main__':
    config=Config()
    train_data=My_Dataset('../data/train.csv',config,1)
    train_iter = DataLoader(train_data, batch_size=32)
    n=0
    for a,b in train_iter:
        b0,b1=b
        n=n+1

        print(n,b0.shape,b1.shape)
        #print(y)
        print('************')