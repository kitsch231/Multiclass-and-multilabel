# coding: UTF-8
import numpy as np
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
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

def train(config, model, train_iter, dev_iter, test_iter,writer):
    start_time = time.time()
    # writer.add_graph(model,input_to_model=((torch.rand(4,256,256,3).to(config.device),
    #                                         torch.LongTensor(4,128).to(config.device),
    #                                         torch.LongTensor(4,128).to(config.device)),))
    model.train()

    if config.mynet=='bert':
        optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if 'bert' in n],'lr': config.bert_learning_rate},#包含bert层学习率
                                        {'params': [p for n, p in model.named_parameters() if 'bert' not in n],'lr':config.other_learning_rate}]
        optimizer = torch.optim.Adam(optimizer_grouped_parameters , lr=config.other_learning_rate)  ## 定义优化器
    else:
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    #optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,10, gamma=0.5, last_epoch=-1)#每2个epoch学习率衰减为原来的一半

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    for epoch in range(config.num_epochs):
        loss_list=[]#承接每个batch的loss
        acc0_list=[]
        acc1_list = []
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            labels0,labels1=labels
            outputs0,outputs1 = model(trains)
            optimizer.zero_grad()
            fcloss0 = FocalLoss(2)
            fcloss1 = FocalLoss(3)
            loss0 = fcloss0(outputs0, labels0)
            loss1 = fcloss1(outputs1, labels1)
            loss=(loss0+loss1)/2
            loss.backward()
            optimizer.step()

            true0 = labels0.data.cpu()
            predic0 = torch.max(outputs0.data, 1)[1].cpu()
            train_acc0 = metrics.accuracy_score(true0, predic0)

            true1 = labels1.data.cpu()
            predic1 = torch.max(outputs1.data, 1)[1].cpu()
            train_acc1 = metrics.accuracy_score(true1, predic1)


            writer.add_scalar('train/loss_iter', loss.item(),total_batch)
            writer.add_scalar('train/acc0_iter',train_acc0,total_batch)
            writer.add_scalar('train/acc1_iter', train_acc1, total_batch)
            msg1 = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc0: {2:>6.2%},  Train Acc1: {3:>6.2%}'
            if total_batch%20==0:
                print(msg1.format(total_batch, loss.item(), train_acc0,train_acc1))
            loss_list.append(loss.item())
            acc0_list.append(train_acc0)
            acc1_list.append(train_acc1)


            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过2000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

        dev_acc0,dev_acc1,dev_loss = evaluate(config, model, dev_iter)#model.eval()
        if dev_loss < dev_best_loss:
            dev_best_loss = dev_loss
            torch.save(model.state_dict(), config.save_path)
            improve = '*'
            last_improve = total_batch
        else:
            improve = ''
        time_dif = get_time_dif(start_time)
        epoch_loss=np.mean(loss_list)
        epoch_acc0=np.mean(acc0_list)
        epoch_acc1 = np.mean(acc1_list)
        msg2 = 'EPOCH: {0:>6},  Train Loss: {1:>5.2},  Train Acc0: {2:>6.2%},  Train Acc1: {3:>6.2%},  Val Loss: {4:>5.2},  Val Acc0: {5:>6.2%},  Val Acc1: {6:>6.2%},  Time: {7} {8}'
        print(msg2.format(epoch+1,epoch_loss, epoch_acc0,epoch_acc1, dev_loss, dev_acc0,dev_acc1, time_dif, improve))
        writer.add_scalar('train/loss_epoch',epoch_loss, epoch)
        writer.add_scalar('train/acc0_epoch', epoch_acc0, epoch)
        writer.add_scalar('train/acc1_epoch', epoch_acc1, epoch)
        writer.add_scalar('val/loss_epoch', dev_loss, epoch)
        writer.add_scalar('val/acc0_epoch', dev_acc0, epoch)
        writer.add_scalar('val/acc1_epoch', dev_acc1, epoch)

        model.train()
        scheduler.step()
        print('epoch: ', epoch+1, 'lr: ', scheduler.get_last_lr())

    #test(config, model, test_iter)


def test(config, model, test_iter):
    # 测试函数
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc0,test_acc1, test_loss, test_report0,test_report1, test_confusion0 ,test_confusion1 = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc0: {1:>6.2%},  Test Acc1: {2:>6.2%}'
    print(msg.format(test_loss, test_acc0,test_acc1))

    print('*********************分割线************************')
    print("标签0：Precision, Recall and F1-Score...") #精确率和召回率以及调和平均数
    print(test_report0)
    print("标签0：Confusion Matrix...")
    print(test_confusion0)

    print('*********************分割线************************')
    print("标签1：Precision, Recall and F1-Score...") #精确率和召回率以及调和平均数
    print(test_report1)
    print("标签1：Confusion Matrix...")
    print(test_confusion1)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all0 = np.array([], dtype=int)
    labels_all0 = np.array([], dtype=int)

    predict_all1 = np.array([], dtype=int)
    labels_all1 = np.array([], dtype=int)

    with torch.no_grad():
        for texts, labels in data_iter:
            #print(texts)
            labels0,labels1=labels
            outputs0,outputs1 = model(texts)
            fcloss0 = FocalLoss(2)
            fcloss1 = FocalLoss(3)
            loss0 = fcloss0(outputs0, labels0)
            loss1 = fcloss1(outputs1, labels1)
            loss = (loss0 + loss1) / 2
            loss_total += loss

            labels0 = labels0.data.cpu().numpy()
            predic0 = torch.max(outputs0.data, 1)[1].cpu().numpy()

            labels1 = labels1.data.cpu().numpy()
            predic1 = torch.max(outputs1.data, 1)[1].cpu().numpy()

            # print(outputs)
            # print(predic)
            # print(labels)
            # print('*************************')
            labels_all0 = np.append(labels_all0, labels0)
            predict_all0 = np.append(predict_all0, predic0)

            labels_all1 = np.append(labels_all1, labels1)
            predict_all1 = np.append(predict_all1, predic1)

    acc0 = metrics.accuracy_score(labels_all0, predict_all0)
    acc1 = metrics.accuracy_score(labels_all1, predict_all1)
    # print(labels0)
    # print(predic0)
    # print(labels1)
    # print(predic1)
    if test:
        report0 = metrics.classification_report(labels_all0, predict_all0, digits=4)
        confusion0 = metrics.confusion_matrix(labels_all0, predict_all0)

        report1 = metrics.classification_report(labels_all1, predict_all1, digits=4)
        confusion1 = metrics.confusion_matrix(labels_all1, predict_all1)
        return acc0,acc1, loss_total / len(data_iter), report0,report1, confusion0,confusion1

    return acc0,acc1, loss_total / len(data_iter)



if __name__ == '__main__':

    config = Config()
    writer = SummaryWriter(log_dir=config.log_dir)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    print("Loading data...")


    train_data=My_Dataset('../data/train.csv',config,1)
    dev_data = My_Dataset('../data/val.csv',config,1)
    test_data = My_Dataset('../data/test.csv',config,1)


    train_iter=DataLoader(train_data, batch_size=config.batch_size,shuffle=True)   ##训练迭代器
    dev_iter = DataLoader(dev_data, batch_size=config.batch_size,shuffle=True)      ###验证迭代器
    test_iter = DataLoader(test_data, batch_size=config.batch_size,shuffle=True)   ###测试迭代器
    # 训练
    if config.mynet=='bert':
        mynet=Mybert(config)
    elif config.mynet=='cnn':
        mynet =Mycnn(config)

    elif config.mynet=='rnn':
        mynet =Myrnn(config)
    ## 模型放入到GPU中去
    mynet= mynet.to(config.device)
    print(mynet.parameters)

    #训练结束后可以注释掉train函数只跑test评估模型性能
    #test(config, mynet, test_iter)
    train(config, mynet, train_iter, dev_iter, test_iter,writer)
    test(config, mynet, test_iter)

#tensorboard --logdir=bert_rnn_cnn/log/bert/1675321719.8882973 --port=6006