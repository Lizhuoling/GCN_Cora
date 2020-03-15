# GCN_Cora
This program tackles the Cora dataset using graph convolutional neural (GCN)

## Introduction
Cora dataset contains more than a thousand papers belonging to seven categories. Meanwhile, they are cited by each other. This dataset contains the word vector information of the papers and their citation relationship, which is actually a graph consisting of many nodes. We aim to classifiy the categories of these papers based on the word vector information and citation information. In this program, we tackle this problem using graph coonvolutional neural network (GCN) proposed in the paper "Semi-supervised classification with Graph Convolutional Networks". 

## Environment requirement
Python2 or Python3
Linux operation system (We use Ubuntu17)
The used libraries should be installed by yourself, which mainly include pytorch, numpy, matplotlib, sklearn, pandas, scipy, etc.

## Running step
You should first use the build.sh to prepare the running environment. You can use the following commands.
```
sudo chmod +x build.sh
./build.sh
```
Then, you can use the train.py to train your model.
```
python train.py
```
The performance can be verified using validation.py.
```
python validation.py
```
