# -*- coding: UTF-8 -*-
"""
    introduction: train a delta inverse kinematics model of hr6
"""
import  torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from meta import Meta
import  argparse
import numpy as np
from models import ann_model
import matplotlib as mpl
import matplotlib.pyplot as plt
from fk_models import *
class DIM():
    def __init__(self,config=None,p_start=[],q_start=[],p_tgt=[]):
        if not config:
            self.config = [
                ('linear', [input_size, 128]),
                ('relu', [True]),
                # ('bn', [128]),

                ('linear', [128, 128]),
                ('relu', [True]),
                # ('bn', [128]),

                ('linear', [128, 128]),
                ('relu', [True]),
                # ('bn', [128]),

                ('linear', [128, 128]),
                ('relu', [True]),
                # ('bn', [128]),

                ('linear', [128, output_size]),
                # ('sigmoid', [True])
            ]
        else:
            self.config = config
        self.DeltaModel = ann_model(config)
        self.q_s = q_start
        self.p_s = p_start
        self.p_t = p_tgt

    
def main(args):
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    print(args)
        #整体模型架构
    config = [
        ('linear', [3, 128]),
        ('relu', [True]),
        # ('bn', [128]),

        ('linear', [128, 128]),
        ('relu', [True]),
        # ('bn', [128]),

        ('linear', [128, 128]),
        ('relu', [True]),
        # ('bn', [128]),

        ('linear', [128, 128]),
        ('relu', [True]),
        # ('bn', [128]),

        ('linear', [128, 6]),
        # ('sigmoid', [True])
    ]
    generate_ = args.generate_data

    q_0 = np.array([97.27,30.76,13.48,-4.10,-3.22,-67.97 ])
    d2r = np.pi/180
    q_0 = np.array([q_i * d2r for q_i in q_0])
    dobot = get_Robot()
    p_0 = dobot.cal_fk(q_0)
    print("p_0 = ")
    print(p_0)
    inputs, outputs, test_set, delta_p_range, delta_q_range = generate_delta_data(q_0, p_0,500)
    test_inputs = np.array(test_set[0])
    test_outputs = np.array(test_set[1])

    test_set = [test_inputs.astype(float), test_outputs.astype(float)]
    inputs = inputs.astype(float)
    outputs = outputs.astype(float)
    print(test_set[0][0],test_set[1].shape)
    print("inputs.shape, outputs.shape ", inputs.shape, outputs.shape)
    print("inputs[0]: ", inputs[0])
    print("outputs[0]: ", outputs[0])
    print("delta_p_range:", delta_p_range)

    #整体模型架构
    

    fk_nn = ann_model(config)

    #损失函数
    cost = torch.nn.MSELoss(reduction='mean')
    #参数
    optimizer_ik = torch.optim.Adam(fk_nn.parameters(), lr = 0.001)
    # 训练网络
    losses = []
    batch_size = args.batch_size
    if args.generate_data:
        dobot = get_Robot()
    losses_train = []
    losses_test = []
    batch_loss = []
    for i in range(args.epoch):
        if i % 5==0:
            with torch.no_grad():
                losses.append(np.mean(batch_loss))
                test_x = torch.tensor(test_set[0], dtype = torch.float, requires_grad = False)
                test_y = torch.tensor(test_set[1], dtype = torch.float, requires_grad = False)
                prediction_ = fk_nn(test_x)
                test_loss = cost(prediction_, test_y)
                losses_test.append(test_loss.data.numpy())
                print(i, " training loss ", np.mean(batch_loss), " test loss ", test_loss.data.numpy())

                rands = np.random.choice(prediction_.data.numpy().shape[0])
                joint_1 = [delta_q_i * delta_q_range[1] + delta_q_range[0] +\
                                q_0 for delta_q_i in prediction_.data.numpy()]
                print("joint[i]:")
                print(joint_1[-1])
                p_pre = np.array([dobot.cal_fk(joint_i) for joint_i in joint_1])
                print("p_pre")
                print(p_pre[-1])
                p_real = [  p_0 + inputs_[0:3] * delta_p_range[1] + delta_p_range[0] \
                            for inputs_ in test_x.data.numpy()]
                print("p_real")
                print(p_real[-1])
                print(p_0)

                dist_0 = []
                dist_1 = []
                for i in range(0,p_pre.shape[0]):
                    tmp = np.array(p_pre[i] - p_real[i], dtype=np.float64)
                    tmp2 = np.array(p_0 - p_real[i], dtype=np.float64)
                    dist_0.append(np.linalg.norm(tmp))
                    dist_1.append(np.linalg.norm(tmp2))
                    # print(dist_1[-1])
                joint_1 = [delta_q_i * delta_q_range[1] + delta_q_range[0] +\
                                q_0 for delta_q_i in prediction_.data.numpy()]
                losses_train.append(dist_0)
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
        # losses_train.append(batch_loss)
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
    
    plt.plot(range(len(losses_train)),loss_mean) 
    plt.xlabel('epoch')
    # plt.ylabel('mse loss')
    plt.ylabel('distance between p\' and p (cm)')
    plt.fill_between(range(len(losses_train)),
                    loss_min,
                    loss_max,
                    color='b',
                    alpha=0.2)
    plt.show()
    torch.save(fk_nn, "./model_trained/delta_net.pkl")


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
    
    args = argparser.parse_args()
    main(args)