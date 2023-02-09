import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
import sys
import torch
import numpy as np
from models import Mybert,Mycnn,Myrnn
from tensorboardX import SummaryWriter
from utils import My_Dataset,get_time_dif
from models import *
from config import Config
from tqdm import tqdm
from torch.utils.data import DataLoader


#用来预测未知数据
config=Config()
test_data = My_Dataset('../data/test.csv',config,0)
test_iter = DataLoader(test_data, batch_size=config.batch_size,shuffle=False)   ###测试迭代器

print(config.mynet)
if config.mynet == 'bert':
    mynet = Mybert(config)
elif config.mynet == 'cnn':
    mynet = Mycnn(config)
elif config.mynet == 'rnn':
    mynet = Myrnn(config)
## 模型放入到GPU中去

all_pre0=[]
all_pre1=[]
mynet = mynet.to(config.device)
print(mynet.parameters)
print(config.save_path)
mynet.load_state_dict(torch.load(config.save_path))
mynet.eval()
with torch.no_grad():
    for texts in test_iter:

        outputs0,outputs1 = mynet(texts)
        predic0 = torch.max(outputs0.data, 1)[1].cpu().numpy()  ###预测结果
        predic1 = torch.max(outputs1.data, 1)[1].cpu().numpy()  ###预测结果

        for x in predic0:
            #print(x)
            all_pre0.append(x)
        for x in predic1:
            #print(x)
            all_pre1.append(x)

# label0=pd.read_csv('../data/test.csv')['label1'].to_list()
# report0 = metrics.classification_report(label0,all_pre0, digits=4)
# print(label0)
# print(report0)