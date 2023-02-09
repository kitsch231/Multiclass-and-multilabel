import pandas as pd
import jieba
import tqdm
'''此代码的作用是读取数据和停用词 进行文本的分词处理和清洗'''

def cdata(path,rpath):
    with open('mldata/stopwords.txt','r',encoding='utf-8')as f:
        stop=f.readlines()
        stop=[x.strip() for x in stop]#去除换行符
    print(stop)

    df=pd.read_csv(path)#读取数据
    df=df.dropna()#删除空行
    print(df)
    texts=df['微博正文'].to_list()
    cut_text=[]#分词后数据

    for text in tqdm.tqdm(texts):
        text=text.replace('\n','')#去除空格
        cut_words=jieba.lcut(text)
        cut_words=[x for x in cut_words if x not in stop]
        cut_text.append(' '.join(cut_words))

    df['cut_text']=cut_text
    df=df.dropna()#删除空行
    df=df[df['cut_text'].str.len()>=3]#剔除掉字符少于3个的数据
    df.to_csv(rpath,index=None)

cdata('../data/test.csv','mldata/mltest.csv')
cdata('../data/train.csv','mldata/mltrain.csv')
cdata('../data/val.csv','mldata/mlval.csv')