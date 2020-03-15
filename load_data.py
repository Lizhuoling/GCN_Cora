#coding:utf-8

from __future__ import division
from __future__ import print_function

import os
import sys
import pandas as pd
import numpy as np

if __name__=='utils.load_data':
    root_path='../'
elif __name__=='__main__':
    root_path=os.path.dirname(os.path.dirname(os.path.abspath('./')))
    if root_path not in sys.path:
        sys.path.insert(0,root_path)

from gcn_cora.utils.defi import defi

class load_data(defi):
    def __init__(self):
        super(load_data,self).__init__()
        self.data_path=root_path+'/gcn_cora/data/'
        self.content_file='cora.content'
        self.cite_file='cora.cites'

        #Pre-defined global variable
        self.index_map=None #mapping original paper index to[0,2707]
        self.features=None  #total data
        self.data=None
        self.label=None
        self.train_label=None
        self.val_label=None
        self.adjacent=None
        self.train_adj=None
        self.val_adj=None

        #initialition function
        self.preprocess_data()

    def preprocess_content(self):
        raw_data=pd.read_csv(self.data_path+self.content_file,sep='\t',header=None) #read content file
        #mapping the original paper index to [0,2707]
        a=list(raw_data.index)
        b=list(raw_data[0])
        c=zip(b,a)
        self.index_map=dict(c)   #mapping the original paper index to [0,2707]
        #extract the features
        self.features=raw_data.iloc[:,1:-1]  #shape:(2708,1433)
        #one-hot encoding the labels
        label=raw_data.iloc[:,-1]
        #self.label=pd.get_dummies(label)
        self.label=self.process_label(label)

    #create the adjacent matrix(+I)
    def preprocess_cite(self):
        raw_cite=pd.read_csv(self.data_path+self.cite_file,sep='\t',header=None)
        self.adjacent=np.zeros((self.paper_num,self.paper_num),dtype=np.float32)
        for i,j in zip(raw_cite[0],raw_cite[1]):
            x=self.index_map[i]
            y=self.index_map[j]
            self.adjacent[x][y]=self.adjacent[y][x]=1
        I=np.identity(self.paper_num,dtype=np.float32)
        self.adjacent+=I
        self.adjacent=self.norm_adj(self.adjacent)

    #split the dataset into training data and validation data
    def split_data(self):
        self.features=np.array(self.features).astype(np.float32)
        self.data=self.norm_data(self.features)
        self.train_label=np.array(self.label.loc[0:self.train_num-1]).astype(np.float32)
        self.val_label=np.array(self.label.loc[self.train_num:self.train_num+self.val_num-1]).astype(np.float32)
        self.train_adj=self.adjacent[0:self.train_num,:]
        self.val_adj=self.adjacent[self.train_num:self.train_num+self.val_num,:]

    def preprocess_data(self):
        self.preprocess_content()
        self.preprocess_cite()
        self.split_data()

    def process_label(self,label):
        label=label.map(lambda x:self.label_map[x])
        return label

    def norm_data(self,x):
        rowsum=np.sum(x,axis=1)
        r_inv=np.power(rowsum,-1).flatten()
        r_inv[np.isinf(r_inv)]=0.
        r_mat_inv=np.diag(r_inv)
        return np.matmul(r_mat_inv,x)

    #compute the D^{-0.5}*A*D^{-0.5} of the paper
    def norm_adj(self,adj):
        if adj.shape[0]!=adj.shape[1]:
            raise Exception('This is not an adjacent matrix, shape[0]!=shape[1]')
        D=np.sum(adj,axis=1)
        D=np.power(D,-0.5).flatten()
        D[np.isinf(D)]=0.
        D=np.diag(D)
        ter=np.matmul(D,adj)
        return np.matmul(adj,D)

    def train_loader(self,batch):
        for i in range(int(self.train_num/batch)):
            label=self.train_label[i*batch:(i+1)*batch]
            adj=self.train_adj[i*batch:(i+1)*batch,:]
            yield self.data,label,adj

    def val_loader(self,batch):
        for i in range(int(self.val_num/batch)):
            label=self.val_label[i*batch:(i+1)*batch]
            adj=self.val_adj[i*batch:(i+1)*batch,:]
            yield self.data,label,adj

if __name__=='__main__':
    model=load_data()
    for data,label,adj in model.train_loader(10):
        sys.exit()
