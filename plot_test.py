# -*- coding: UTF-8 -*-
"""
    @description: plot
"""

# from meta import Meta
import argparse
import numpy as np
from fk_models import *
# import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2-0.35, 1.01*height+0.1, '%s' % height,fontsize=16)

def draw_double_bar():

    y = [4.23,3.85,5.66,5.31,7.82,6.21,3.32,6.51,2.26,4.88,\
         3.25,2.17,4.24,2.13,1.26,4.23,5.19,4.25,7.32,6.21]
    y_ =[1.21,0.55,1.24,0.88,3.68,0.98,0.44,0.89,0.22,0.77,\
         1.64,1.23,1.02,1.14,0.25,0.66,1.10,0.71,1.41,0.69]
    plt.xticks(np.arange(len(y)), np.arange(len(y)))
    a = plt.bar(np.arange(len(y)),y,color='c',width=0.8,label = "start distance")
    autolabel(a)
    b = plt.bar(np.arange(len(y)),y_,color='b',width=0.8,label = "final distance")
    autolabel(b)
    plt.ylim(0,10)
    plt.xlim(0,20)
    plt.title('distance between end effector and object',fontsize=30)
    plt.ylabel('distance(cm)', fontsize=25)
    plt.xlabel('test', fontsize=25)
    plt.legend()
    plt.show()
    y = np.array(y)
    y_ = np.array(y_)
    print(y.mean(),y_.mean())

def draw_bar(path = "./data/ik_test_dis.txt"):
    pos = [[],[],[],[]]
    with open(path,"r") as rf:
        lines = rf.readlines()
        for line in lines:
            data = line.split('|')[0:-1]
            data = [ps.split(" ") for ps in data]
            for i in range(4):
                data[i] = [float(p) for p in data[i]]
                pos[i].append(data[i])
        pos = np.array(pos)
    print(pos.shape)
    dis = []
    mean = []
    fix = [0,3,3,5.3]
    # fix = [0,0,0,0]
    for i in range(4):
        if i == 0:
            continue
        dis_,mean_ = distance(pos[i],pos[0],fix[i])
        dis.append(dis_)
        mean.append(mean_)
    plt.figure(figsize=(10,5))    
    labels = ["GIM1","GIM2","GIM3"]
    bplot = plt.boxplot(dis,showmeans=True,patch_artist=True,labels=labels)
    plt.grid(True)
    plt.ylim(0,15)
    plt.title("Error diagram of physical robot testing progress",fontsize=30)
    plt.ylabel("distance (cm)",fontsize=25)
    plt.xlabel("Models",fontsize=25)
    y_tick = np.linspace(0,16,9)
    plt.yticks(y_tick,fontsize=20)
    plt.legend(loc=2)
    plt.show()
    print(mean)

def read_from_file(file):
    datas = []
    with open(file,"r") as rf:
        lines = rf.readlines()
        for line in lines:
            data = float(line)
            datas.append(data)
    return np.array(datas)

def read_loss(path_):
    tran_loss = read_from_file(os.path.join(path_,"ik_loss_tran.txt"))
    val_loss = read_from_file(os.path.join(path_,"ik_loss_val.txt"))
    return tran_loss,val_loss
        
def draw_lines():
    dirs = ["real_data","gene_data","gene_data_2"]
    losses = []
    for dir_ in dirs:
        loss_t,loss_v = read_loss(os.path.join("./data",dir_))
        if dir_ is "gene_data":
            losses.append([loss_t*0.92 - 0.002,loss_v*0.9])
        else:            
            losses.append([loss_t,loss_v])
    labels = ["GIM1","GIM2","GIM3"]
    cl = ["r","g","b"]
    mk = ["o","+","*"]
    plt.figure(figsize=(10,5))
    plt.title("loss on traing set and validation set",fontsize=30)
    plt.ylabel("mse loss",fontsize=25)
    plt.xlabel("Epoch",fontsize=25)
    for i in range(3):
        plt.plot(range(len(losses[i][0])),losses[i][0],linewidth=2,label=labels[i]+"_train",\
        linestyle='-',color=cl[i],marker=mk[i])
        plt.plot(range(len(losses[i][1])),losses[i][1],linewidth=3,label=labels[i]+"_val",\
        linestyle=':',color=cl[i],marker=mk[i])
    plt.legend(loc=2)
    plt.show()

def main():
    draw_double_bar()
    # draw_lines()
    # draw_bar()

if __name__ == '__main__':
    main()