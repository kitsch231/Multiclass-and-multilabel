import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression,Lasso
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay,accuracy_score
import pickle
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from matplotlib import pyplot as plt

df=pd.read_csv('mldata/mltrain.csv')
testdf=pd.read_csv('mldata/mltest.csv')
# print(df)
texts=df['cut_text'].to_list()#读取分好词的句子
labels=df[['label0','label1']]#读取标签
labels=np.array(labels)
#max_features=2000
vectorizer = CountVectorizer(decode_error="replace",max_features=5000)#最大特征词数
tfidftransformer = TfidfTransformer()
# 注意在训练的时候必须用vectorizer.fit_transform、tfidftransformer.fit_transform
# 在预测的时候必须用vectorizer.transform、tfidftransformer.transform
vec_train = vectorizer.fit_transform(texts)#转换好的词频矩阵
tfidf = tfidftransformer.fit_transform(vec_train)#词频矩阵转换成tfidf矩阵

train_x=tfidf
train_y=labels
test_x=tfidftransformer.transform(vectorizer.transform(testdf['cut_text'].to_list()))
test_y=testdf[['label0','label1']]
test_y=np.array(test_y)
# test_x=tfidf
# test_y=labels

# 保存经过fit的vectorizer 与 经过fit的tfidftransformer,预测时使用
feature_path = 'models/feature.pkl'
with open(feature_path, 'wb') as fw:
    pickle.dump(vectorizer.vocabulary_, fw)

tfidftransformer_path = 'models/tfidftransformer.pkl'
with open(tfidftransformer_path, 'wb') as fw:
    pickle.dump(tfidftransformer, fw)
print('tfidf:',tfidf.shape)#查看tfidf的shape大小

# #划分数据集，测试集20%，其中random_state为随机种子固定一个数以免每次划分的数据集不一样导致结果不一样
# train_x,test_x,train_y,test_y= train_test_split(tfidf,labels,test_size=0.3,random_state=5)

def get_acc(name,test_y,rf_pre):
    print('准确率是:', accuracy_score(test_y, rf_pre))
    print(classification_report(test_y, rf_pre))
    rf_confusion = confusion_matrix(test_y, rf_pre)  # 计算混淆矩阵
    rf_disp = ConfusionMatrixDisplay(confusion_matrix=rf_confusion)  # 可视化混淆矩阵
    rf_disp.plot(
        include_values=True,  # 混淆矩阵每个单元格上显示具体数值
        cmap="viridis",
        ax=None,
        xticks_rotation="horizontal",
        values_format="d"  # 显示的数值格式
    )
    plt.savefig(name+'rf混淆矩阵.png')
    plt.show()

print('随机森林........')
rfmodel=RandomForestClassifier()
rfmodel=MultiOutputClassifier(rfmodel)
rfmodel.fit(train_x,train_y)
rf_pre=rfmodel.predict(test_x)
#打印模型报告
print('相关性报告:')
get_acc('相关性',test_y[:,0],rf_pre[:,0])
print('*************分割线***************')

print('倾向性报告:')
get_acc('倾向性',test_y[:,1],rf_pre[:,1])
print('*************分割线***************')