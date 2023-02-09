'''数据标签重新整理并划分数据集'''
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

path='alldata.csv'
df=pd.read_csv(path,encoding='gb18030')
print(df)

def clabel(df,pw,nw,lw):

    labels=[]
    for x in range(len(df)):
        d1=df.iloc[x,:]
        #print(d1)
        p=d1[pw]
        n=d1[nw]
        #print(p,n)
        if p==1:
            labels.append(2)
        elif n==1:
            labels.append(0)
        else:
            labels.append(1)
    df[lw]=labels
    return df

df=clabel(df,'正面','负面','label1')
df['label0']=df['相关']


train,_=train_test_split(df,test_size=0.3,random_state=10)
val,test=train_test_split(_,test_size=0.5,random_state=10)
train.to_csv('data/train.csv',index=None)
val.to_csv('data/val.csv',index=None)
test.to_csv('data/test.csv',index=None)


wordnum=df['微博正文'].str.len().to_list()
print(np.mean(wordnum),'平均长度')