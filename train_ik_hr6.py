# -*- coding: UTF-8 -*-
"""
    introduction: train a fk model of hr6
"""
import  torch
from torch import nn
from torch import optim
from torch.nn import functional as F
# from meta import Meta
import  argparse
import numpy as np
from fk_models import *
from models import *
# import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import train_fk_hr6
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
import time
#整体模型架构
config = [
    ('linear', [3, 64]),
    ('relu', [True]),
    # ('bn', [64]),

    ('linear', [64, 64]),
    ('relu', [True]),
    # # ('bn', [128]),

    # # # ('linear', [256, 128]),
    # # # ('relu', [True]),
    # # # # ('bn', [128]),

    # ('linear', [64, 32]),
    # ('relu', [True]),
    # # ('bn', [32]),
    # ('linear', [32, 32]),
    # ('relu', [True]),
    ('linear', [64, 6]),
    # ('sigmoid', [True])
]
def read_min_max(path):
    q_range = []
    p_range = []
    with open(path,"r") as rf:
        line = rf.readline()
        line = line.strip().split(",")
        line = [float(num) for num in line]
        q_range.append(line[0:6])
        q_range.append(line[6:])
        line = rf.readline()
        line = line.strip().split(",")
        line = [float(num) for num in line]
        p_range.append(line[0:3])
        p_range.append(line[3:])
    p_range = np.array(p_range)
    q_range = np.array(q_range)
    return p_range,q_range

p_range, q_range = read_min_max("./model_trained/min_max.txt")

def main(args):
    global config,p_range, q_range
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    print(args)
        #整体模型架构
    generate_ = args.g
    is_delta_model = args.is_delta_model
    fk_nn = ann_model(train_fk_hr6.config)
    fk_nn.load_state_dict(torch.load("./model_trained/fk_net_param.pkl"))

    if generate_:
        inputs, outputs, test_inputs, test_outputs, _, _ = generate_data(1000, is_fk=args.is_fk)
    else:
        inputs, outputs, test_inputs, test_outputs, _, _ = load_data("./data/"+args.file+".txt",training_nums=args.n)
    test_set = [test_inputs.astype(float), test_outputs.astype(float)]
    inputs = inputs.astype(float)
    outputs = outputs.astype(float)
    print(test_set[0][0],test_set[1].shape)
    print("inputs.shape, outputs.shape ", inputs.shape, outputs.shape)
    print("inputs[0]: ", inputs[0])
    print("outputs[0]: ", outputs[0])
    # print("p_range: ", p_range)
    # print("q_range: ", q_range)
    input_size = inputs.shape[1]
    hidden_size = args.hidden_size
    hidden_layer = args.hidden_layer
    output_size = outputs.shape[1]

    ik_nn = ann_model(config)
    if args.l:
        print("load pretrain models!")
        ik_nn.load_state_dict(torch.load("./model_trained/net_param.pkl"))
        # print(ik_nn.state_dict())
    if args.m:
        print("load meta models!")
        ik_nn.load_state_dict(torch.load("./model_trained/meta_ik_rjoint_2.pkl",map_location=torch.device('cpu')))
    #损失函数
    cost = torch.nn.MSELoss(reduction='mean')
    #参数
    if args.l2:
        optimizer_ik = torch.optim.Adam(ik_nn.parameters(), lr = 0.001,betas=(0.9,0.999),weight_decay=0.001)
    else:
        optimizer_ik = torch.optim.Adam(ik_nn.parameters(), lr = 0.001,betas=(0.9,0.999))
    # 训练网络
    losses_train = []
    batch_size = args.batch_size
    # pku_hr6 = get_Robot()
    links_len = np.array([-1.63, -20,12,11,10, -1,-3,0.68,])
    pku_hr6 =  get_Robot_rand(links_len)
    dis_test = []
    dis_train = []
    losses_test = []
    test_x = torch.tensor(test_set[0], dtype = torch.float, requires_grad = False)
    test_y = torch.tensor(test_set[1], dtype = torch.float, requires_grad = False)
    with torch.no_grad():
        prediction_ = ik_nn(test_x)
        test_loss = cost(prediction_, test_y)
        losses_test.append(test_loss.data.numpy())
        p_pre_nn = fk_nn(prediction_)
        p_pre_nn = np.array([inputs_[0:3]*p_range[1] + p_range[0] for inputs_ in p_pre_nn.data.numpy()])
        p_real = np.array([inputs_[0:3]*p_range[1] + p_range[0] for inputs_ in test_x.data.numpy()])
        dis = p_pre_nn - p_real
        dis = np.array([np.linalg.norm(d) for d in dis])
        dis_test.append(dis)

        train_x = torch.tensor(inputs, dtype = torch.float, requires_grad = False)
        train_y = torch.tensor(outputs, dtype = torch.float, requires_grad = False)
        prediction_ = ik_nn(train_x)
        test_loss = cost(prediction_, train_y)
        losses_train.append(test_loss.data.numpy())
        p_pre_nn = fk_nn(prediction_)
        p_pre_nn = np.array([inputs_[0:3]*p_range[1] + p_range[0] for inputs_ in p_pre_nn.data.numpy()])
        p_real = np.array([inputs_[0:3]*p_range[1] + p_range[0] for inputs_ in train_x.data.numpy()])
        dis = p_pre_nn - p_real
        dis = np.array([np.linalg.norm(d) for d in dis])
        dis_train.append(dis)

    for i in range(args.epoch):
        if i % 1==0:
            # for test
            with torch.no_grad():
                if args.d:
                    prediction_ = ik_nn(test_x)
                    test_loss = cost(prediction_, test_y)
                    losses_test.append(test_loss.data.numpy())
                    print("************************* epoch ",i,"*************************")
                    print("some datas:")
                    print("inputs:")
                    ii = test_x.data.numpy()[-1]
                    print(ii)
                    ii = ii*p_range[1] + p_range[0]
                    print(ii)
                    print("outputs:",prediction_.data.numpy()[-1])
                    print("test_real",test_set[1][-1])
                    joint_real = (test_set[1][-1] * q_range[1] + q_range[0])
                    joint_out = [q_i * q_range[1] + q_range[0] for q_i in prediction_.data.numpy()]
                    print("joint real:", joint_real/np.pi * 180)
                    print("joint out:", joint_out[-1]/np.pi * 180)
                    p_pre_fk = pku_hr6.cal_fk(joint_out[-1])
                    p_real_fk = pku_hr6.cal_fk(joint_real)
                    p_pre_nn = fk_nn(prediction_)
                    p_pre_nn = np.array([inputs_[0:3]*p_range[1] + p_range[0] for inputs_ in p_pre_nn.data.numpy()])
                    p_real = np.array([inputs_[0:3]*p_range[1] + p_range[0] for inputs_ in test_x.data.numpy()])
                    print("p_real",p_real[-1])
                    print("p_real_fk",p_real_fk)
                    print("p_pre_1",p_pre_fk)
                    print("p_pre_2",p_pre_nn[-1])
                    dis = p_pre_nn - p_real
                    dis = np.array([np.linalg.norm(d) for d in dis])
                    print("mean dis nn:",np.array(dis).mean())
                    dis_test.append(dis)

                    train_x = torch.tensor(inputs, dtype = torch.float, requires_grad = False)
                    train_y = torch.tensor(outputs, dtype = torch.float, requires_grad = False)
                    prediction_ = ik_nn(train_x)
                    print(train_y.shape)
                    print(prediction_.shape)
                    train_loss = cost(prediction_, train_y)
                    losses_train.append(train_loss.data.numpy())
                    p_pre_nn = fk_nn(prediction_)
                    p_pre_nn = np.array([inputs_[0:3]*p_range[1] + p_range[0] for inputs_ in p_pre_nn.data.numpy()])
                    p_real = np.array([inputs_[0:3]*p_range[1] + p_range[0] for inputs_ in train_x.data.numpy()])
                    dis = p_pre_nn - p_real
                    dis = np.array([np.linalg.norm(d) for d in dis])
                    dis_train.append(dis)
                    print(i, " training loss ", train_loss.data.numpy(), " test loss ", test_loss.data.numpy())
                    print("**********************************************************")


                    
        batch_loss = []
        # MINI-Batch方法来进行训练
        for start in range(0, len(inputs), batch_size):
            end = start + batch_size if start + batch_size < len(inputs) else len(inputs)
            xx = torch.tensor(inputs[start:end], dtype = torch.float, requires_grad = True)
            yy = torch.tensor(outputs[start:end], dtype = torch.float, requires_grad = True)
            # rand_index = np.random.randint(0, len(inputs), min(len(inputs),batch_size))
            # xx = torch.tensor(inputs[rand_index], dtype = torch.float, requires_grad = True)
            # yy = torch.tensor(outputs[rand_index], dtype = torch.float, requires_grad = True)
            # print("xx shape is ",xx.shape)
            prediction = ik_nn(xx)
            loss = cost(prediction, yy)
            batch_loss.append(loss.data.numpy())
            optimizer_ik.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_ik.step()

        # dis_test.append(batch_loss)
    with open("./logs/ik/ik_loss_val_"+args.name+".txt","w") as wf:
        for loss in losses_test:
            wf.write(str(loss)+'\n')
    with open("./logs/ik/ik_loss_train_"+args.name+".txt","w") as wf:
        for loss in losses_train:
            wf.write(str(loss)+'\n')
    if args.d:
        dis_min = []
        dis_max = []
        dis_mean = []
        for i in range(len(dis_train)):
            loss_i = np.array(dis_train[i])
            dis_min.append(np.percentile(loss_i,25))
            dis_max.append(np.percentile(loss_i,75))
            dis_mean.append(loss_i.mean())
        plt.plot(range(len(dis_train)),dis_mean,color='g',label=u"训练集")
        with open("./logs/ik/ik_dis_train_"+args.name+".txt","w") as wf:
            for d_min,d_max,d_mean in zip(dis_min,dis_max,dis_mean):
                wf.write(str(d_min)+","+str(d_max)+","+str(d_mean)+'\n')
        dis_min = []
        dis_max = []
        dis_mean = []
        for i in range(len(dis_test)):
            loss_i = np.array(dis_test[i])
            dis_min.append(np.percentile(loss_i,25))
            dis_max.append(np.percentile(loss_i,75))
            dis_mean.append(loss_i.mean())
        with open("./logs/ik/ik_dis_test_"+args.name+".txt","w") as wf:
            for d_min,d_max,d_mean in zip(dis_min,dis_max,dis_mean):
                wf.write(str(d_min)+","+str(d_max)+","+str(d_mean)+'\n')
        # plt.legend(loc='lower right', frameon=False) # 图例在图形里面
        # plt.legend(loc=8, frameon=False, bbox_to_anchor=(0.5,-0.3))# 图例在图形外面
        print("mean dis:")
        print(dis_mean[-1])
        plt.plot(range(len(dis_test)),dis_mean,color='r',label=u"测试集")
        plt.xlabel(u"迭代次数")
        # plt.ylabel('mse loss')
        plt.ylabel(u'反向模型精度 (cm)')
        plt.fill_between(range(len(dis_test)),
                        dis_min,
                        dis_max,
                        color='b',
                        alpha=0.2,label=u"测试集误差区间")
        plt.legend(loc='upper right', fontsize=10)
        plt.show()
        plt.clf()
        plt.plot(range(len(losses_test)),losses_test,label=u"测试集",color='r')
        plt.plot(range(len(losses_train)),losses_train,label=u"训练集",color='g')
        plt.xlabel(u"迭代次数")
        plt.ylabel(u"损失函数值")
        plt.legend(loc='upper right', fontsize=10)
        plt.show()
    if args.s:
        PATH = "./model_trained/net_param.pkl"
        torch.save(ik_nn.state_dict(), PATH)
        print("save model in",PATH)
        torch.save(ik_nn, "./model_trained/net.pkl")
    # with open("./model_trained/min_max_"+args.file+".txt","w") as wf:
    #     for min_q in q_range[0]:
    #         wf.write(str(min_q)+",")
    #     for i in range(6):
    #         wf.write(str(q_range[1][i]))
    #         if i != 5:
    #             wf.write(",")
    #         else:
    #             wf.write("\n")
    #     for min_p in p_range[0]:
    #         wf.write(str(min_p)+",")
    #     for i in range(3):
    #         wf.write(str(p_range[1][i]))
    #         if i != 2:
    #             wf.write(",")
    #         else:
    #             wf.write("\n")
    position_tgt = [33.68, -14.7283, 6.9515]
    from train_delta_ik_hr6 import get_ik_res
    joint_ik = get_ik_res(ik_nn,position_tgt,q_range,p_range)
    position_cur = pku_hr6.cal_fk(joint_ik)
    print("tgt_p,cur p :",position_tgt,position_cur,cal_dis(position_tgt,position_cur))
    
    # start = time.clock()
    # p_i =  torch.tensor(inputs[-1], dtype = torch.float, requires_grad = True)
    # q_i = ik_nn(p_i)
    # end = time.clock()
    # print('Running time: %s Seconds'%(end-start))
    


if __name__ == "__main__":
    start = time.clock()
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=200)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, \
                                help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, \
                                help='update steps for finetunning', default=10)
    argparser.add_argument('--hidden_size', type=int, help='update steps for finetunning', default=128)
    argparser.add_argument('--hidden_layer', type=int, help='update steps for finetunning', default=3)
    argparser.add_argument('--batch_size', type=int, help='update steps for finetunning', default=40)
    argparser.add_argument('--is_fk', type=bool, help='if is foward', default=False)
    argparser.add_argument('--g', type=bool, \
                                help='generate radom datas or using true data', default=False)
    argparser.add_argument('--file', type=str, \
                            help='file name', default="real_data")
    argparser.add_argument('--name', type=str, \
                            help='file name', default="test")
    argparser.add_argument('--is_delta_model', type=bool, \
                                help='if needs to train a delta_p -> delta_q models', default=False)
    argparser.add_argument('--d', type=bool, \
                                help='if need draw pic', default=True)
    argparser.add_argument('--l', type=bool, \
                            help='if load model', default=False)
    argparser.add_argument('--s', type=bool, \
                            help='if load model', default=False)
    argparser.add_argument('--m', type=bool, \
                        help='if meta model', default=False)
    argparser.add_argument('--l2', type=bool, \
                        help='if l2 loss', default=False)
    argparser.add_argument('--n', type=int, \
                            help='nums of training date', default=500)
    args = argparser.parse_args()
    main(args)
    end = time.clock()
    print('Running time: %s Seconds'%(end-start))