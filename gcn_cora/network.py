#coding:utf-8

from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import torch
from torch import nn

root_path=os.path.dirname(os.path.dirname(os.path.abspath('./')))
if root_path not in sys.path:
    sys.path.insert(0,root_path)

from gcn_cora.utils.defi import defi

class AX(nn.Module):
    def __init__(self):
        super(AX,self).__init__()

    def forward(self,a,x):
        return torch.mm(a,x)    #return size:(paper_num,paper_num)

class AXW(nn.Module):
    def __init__(self,in_nodes,out_nodes):
        super(AXW,self).__init__()
        self.ax=AX()
        self.fc=nn.Linear(in_features=in_nodes,out_features=out_nodes)

    def forward(self,a,x):
        ax=self.ax(a,x)
        axw=self.fc(ax)
        return axw

class XW(nn.Module):
    def __init__(self,in_nodes,out_nodes):
        super(XW,self).__init__()
        self.fc=nn.Linear(in_features=in_nodes,out_features=out_nodes)

    def forward(self,x):
        return self.fc(x)

class network(nn.Module):
    def __init__(self,in_nodes,out_nodes1,out_nodes2):
        super(network,self).__init__()
        self.axw1=AXW(in_nodes,out_nodes1)
        self.xw2=XW(out_nodes1,out_nodes2)

    def forward(self,a,x):
        f1=self.axw1(a,x)
        relu1=torch.nn.functional.relu(f1)
        f2=self.xw2(relu1)
        y=torch.nn.functional.log_softmax(f2,dim=1)
        return y

if __name__=='__main__':
    model=network(2708,200,7)

