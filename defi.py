#coding:utf-8

#The global definition is stated here

class defi(object):
    def __init__(self):
        self.paper_num=2708 #the total number of papers
        self.label_num=7
        self.feature_len=1433
        self.train_num=2000
        self.val_num=708
        self.label_map={
                'Case_Based':0,
                'Genetic_Algorithms':1,
                'Neural_Networks':2,
                'Probabilistic_Methods':3,
                'Reinforcement_Learning':4,
                'Rule_Learning':5,
                'Theory':6}
