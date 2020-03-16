#coding:utf-8

from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import torch
from torch import nn
import argparse
import time

root_path=os.path.dirname(os.path.abspath('./'))
if root_path not in sys.path:
    sys.path.insert(0,root_path)

from utils.defi import defi
from utils.network import network
from utils.load_data import load_data

parser=argparse.ArgumentParser()
parser.add_argument('-iteration',type=int,help='the iterations of training',default=300)
parser.add_argument('-lr',type=float,help='the learning rate',default=1e-2)
parser.add_argument('-batch',type=int,help='the number of samples in a batch',default=10)
parser.add_argument('-model',type=str,help='the name of the model saved',default='gcn.pkl')
args=parser.parse_args()

defier=defi()
paper_num=defier.paper_num
label_num=defier.label_num
feature_len=defier.feature_len
train_num=defier.train_num
val_num=defier.val_num

def train():
    epoches=args.iteration
    lr=args.lr
    batch=args.batch
    model_name=args.model

    model=network(feature_len,100,label_num)
    op=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.5)
    dataloader=load_data()
    loss_value=np.zeros((int(val_num/batch),),dtype=np.float32)
    print('start training')
    start_time=time.time()
    for epoch in range(epoches):
        cnt=0
        for data,label,adj in dataloader.train_loader(batch):
            data=torch.from_numpy(data)
            label=torch.LongTensor(label)
            adj=torch.from_numpy(adj)
            output=model(adj,data)
            loss=torch.nn.functional.nll_loss(output,label)
            op.zero_grad()
            loss.backward()
            op.step()
        for data,label,adj in dataloader.val_loader(batch):
            data=torch.from_numpy(data)
            label=torch.LongTensor(label)
            adj=torch.from_numpy(adj)
            output=model(adj,data)
            loss=torch.nn.functional.nll_loss(output,label)
            loss_value[cnt]=loss
            cnt+=1
        now_time=time.time()
        print('epoch:',epoch+1,'time:',now_time-start_time,'loss:',np.mean(loss_value))
    torch.save(model,'./model/'+model_name)

if __name__=='__main__':
    train()
