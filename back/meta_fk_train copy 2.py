# -*- coding: UTF-8 -*-
# @description: traning a forward model of a 4 DOF robot arm
import  torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from meta import Meta
import  argparse
import numpy as np
from forward_kinematic import *
from fk_model import fk_model
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def main(args):
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    print(args)
        #整体模型架构
    generate_ = args.generate_data
    is_delta_model = args.is_delta_model
    if not is_delta_model:
        if generate_:
            inputs, outputs, test_inputs, test_outputs = generate_data(1000, is_fk=args.is_fk)
        else:
            inputs, outputs, test_inputs, test_outputs = load_data("real_data.txt", is_fk=args.is_fk)
    else:
        q_0 = np.array([1.33,-1.25,0.25,0.29])
        dobot = get_Dobot()
        DH_dobot = dobot.fk(q_0)
        # print(DH_dobot)
        p_0 = np.array(DH_dobot[:,-1][0:3] * 100)
        inputs, outputs, test_set, delta_p_range, delta_q_range = generate_delta_data(q_0, p_0)
        test_inputs = np.array(test_set[0])
        test_outputs = np.array(test_set[1])
    test_set = [test_inputs, test_outputs]
    print("inputs.shape, outputs.shape ", inputs.shape, outputs.shape)
    print("inputs[0]: ", inputs[0])
    print("outputs[0]: ", outputs[0])
    input_size = inputs.shape[1]
    hidden_size = args.hidden_size
    hidden_layer = args.hidden_layer
    output_size = outputs.shape[1]

    #整体模型架构
    config = [
        ('linear', [input_size, 128]),
        ('relu', [True]),
        # ('bn', [128]),

        ('linear', [128, 128]),
        ('relu', [True]),
        # ('bn', [256]),
        ('linear', [128, 128]),
        ('relu', [True]),
        ('linear', [128, 128]),
        ('relu', [True]),

        ('linear', [128, output_size]),
        # ('sigmoid', [True])
    ]

    fk_nn = fk_model(config)

    #损失函数
    cost = torch.nn.MSELoss(reduction='mean')
    #参数
    optimizer_ik = torch.optim.Adam(fk_nn.parameters(), lr = 0.001)
    # 训练网络
    losses = []
    batch_size = args.batch_size
    if args.generate_data:
        dobot = get_Dobot()
    losses_train = []
    losses_test = []
    batch_loss = []
    for i in range(args.epoch):
        if i % 10==0:
            with torch.no_grad():
                losses.append(np.mean(batch_loss))
                test_x = torch.tensor(test_set[0], dtype = torch.float, requires_grad = False)
                test_y = torch.tensor(test_set[1], dtype = torch.float, requires_grad = False)
                prediction_ = fk_nn(test_x)
                test_loss = cost(prediction_, test_y)
                losses_test.append(test_loss.data.numpy())
                print(i, " training loss ", np.mean(batch_loss), " test loss ", test_loss.data.numpy())
                if is_delta_model:
                    rands = np.random.choice(prediction_.data.numpy().shape[0])
                    joint_1 = [delta_q_i * delta_q_range[1] + delta_q_range[0] +\
                                 q_0 for delta_q_i in prediction_.data.numpy()]
                    p_pre = np.array([dobot.fk(joint_i)[:,-1][0:3] * 100 for joint_i in joint_1])
                    p_real = [  p_0 + inputs_[0:3] * delta_p_range[1] + delta_p_range[0] \
                                for inputs_ in test_x.data.numpy()]

                    dist_0 = []
                    dist_1 = []
                    for i in range(0,p_pre.shape[0]):
                        dist_0.append(np.linalg.norm(p_pre[i] - p_real[i]))
                        dist_1.append(np.linalg.norm(p_real[i] - p_0))
                    dist_0 = np.mean(np.array(dist_0))
                    dist_1 = np.mean(np.array(dist_1))
                    print(" mean distance ", dist_0, " distance_2 ", dist_1)
        batch_loss = []
        # MINI-Batch方法来进行训练
        for start in range(0, len(inputs), batch_size):
            end = start + batch_size if start + batch_size < len(inputs) else len(inputs)
            xx = torch.tensor(inputs[start:end], dtype = torch.float, requires_grad = True)
            yy = torch.tensor(outputs[start:end], dtype = torch.float, requires_grad = True)
            prediction = fk_nn(xx)
            loss = cost(prediction, yy)
            batch_loss.append(loss.data.numpy())

            optimizer_ik.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_ik.step()
        losses_train.append(batch_loss)
    loss_min = []
    loss_max = []
    loss_mean = []
    for i in range(len(losses_train)):
        loss_i = np.array(losses_train[i])
        loss_min.append(loss_i.min())
        loss_max.append(loss_i.max())
        loss_mean.append(loss_i.mean())
 
    # plt.legend(loc='lower right', frameon=False) # 图例在图形里面
    # plt.legend(loc=8, frameon=False, bbox_to_anchor=(0.5,-0.3))# 图例在图形外面
    
    plt.plot(range(len(losses_train)),loss_mean) #50条数据不能错
    
    plt.fill_between(range(len(losses_train)),
                    loss_min,
                    loss_max,
                    color='b',
                    alpha=0.2)
    plt.show()
    torch.save(fk_nn, "./model_trained/net.pkl")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=100)
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
    argparser.add_argument('--generate_data', type=bool, \
                                help='generate radom datas or using true data', default=True)
    argparser.add_argument('--is_delta_model', type=bool, \
                                help='if needs to train a delta_p -> delta_q models', default=False)
    
    args = argparser.parse_args()
    main(args)