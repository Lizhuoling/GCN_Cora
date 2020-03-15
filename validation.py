#coding:utf-8
#This file visualizes the results on the test dataset using t-SNE

from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
from sklearn.manifold import TSNE
import argparse
import matplotlib.pyplot as plt
import torch
import copy

root_path=os.path.dirname(os.path.abspath('./'))
if root_path not in sys.path:
    sys.path.insert(0,root_path)

from utils.defi import defi
from utils.network import network
from utils.load_data import load_data

parser=argparse.ArgumentParser()
parser.add_argument('-batch',type=int,help='the number of samples in a batch',default=10)
parser.add_argument('-model',type=str,help='the name of the model saved',default='gcn.pkl')
args=parser.parse_args()

defier=defi()
paper_num=defier.paper_num
label_num=defier.label_num
feature_len=defier.feature_len
train_num=defier.train_num
val_num=defier.val_num
label_map=defier.label_map

def validation():
    batch=args.batch
    model_name=args.model

    model=torch.load('./model/'+model_name)
    dataloader=load_data()
    result=None
    label_list=[]
    for data,label,adj in dataloader.val_loader(batch):
        label_list+=label.tolist()
        data=torch.from_numpy(data)
        label=torch.LongTensor(label)
        adj=torch.from_numpy(adj)
        output=model(adj,data).detach().numpy()
        if result is None:
            result=copy.deepcopy(output)
        else:
            result=np.concatenate((result,output),axis=0)

    #summary
    max_index=np.argmax(result,axis=1)
    total_num=np.zeros((label_num,),dtype=np.int32)
    right_num=np.zeros((label_num,),dtype=np.int32)
    for i in range(max_index.shape[0]):
        ter=int(label_list[i])
        total_num[ter]+=1
        if label_list[i]==max_index[i]:
            right_num[ter]+=1
    for i in range(label_num):
        print(list(label_map.keys())[i],'   ','Total number:',total_num[i],'   ','Right number:',right_num[i])
    #visualization
    ts=TSNE(n_components=2)
    result=ts.fit_transform(result)
    x,y=np.split(result,2,axis=1)
    x=x.reshape(x.shape[0],)
    y=y.reshape(y.shape[0],)
    x_list=[]
    y_list=[]
    for i in range(label_num):
        x_list.append([])
        y_list.append([])
    for i in range(len(label_list)):
        ter=int(label_list[i])
        x_list[ter].append(x[i])
        y_list[ter].append(y[i])
    color=[]
    for i in range(label_num):
        color.append([])
        ter=[i/label_num,1-i/label_num,1-i/label_num]
        for j in range(len(x_list[i])):
            color[i].append(ter)
    plt.figure(figsize=(10,6))
    for i in range(label_num):
        plt.scatter(x_list[i],y_list[i],c=color[i],label=list(label_map.keys())[i],zorder=i)
    plt.legend()
    plt.show()

if __name__=='__main__':
    validation()

