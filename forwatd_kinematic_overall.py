# -*- coding: UTF-8 -*-
"""
    @description: overall system of model training
"""
import numpy as np
import math
from functools import reduce
np.set_printoptions(precision=4, suppress=True)
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision import datasets, transforms
from torch.nn import init
import argparse
from random import shuffle


class sinNN(nn.Module):
    def __init__(self, input_size):
        super(sinNN, self).__init__()   # nn.Module子类的函数必须在构造函数中执行父类的构造函数
		# self.weights = nn.Parameter(torch.Tensor(emb_size, emb_size), requires_grad=requires_grad)
		# self.weights = nn.Parameter(torch.Tensor(input_size, input_size*2), requires_grad=requires_grad)
        # if bias:
        #     self.bias = nn.Parameter(torch.Tensor(output_features))
        # else:
        #     # You should always register all possible parameters, but the
        #     # optional ones can be None if you want.
        #     self.register_parameter('bias', None)
		# 初始化参数
    def forward(self, inputs):
        outputs_1 = torch.sin(inputs)
        outputs_2 = torch.cos(inputs)
        inputs = torch.cat((outputs_1, outputs_2), dim=1)
        # print("after cos\sin , shape is ",inputs.shape)
        return inputs

class fk_model(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, output_size):
        super(fk_model, self).__init__()
        lys = []
        # self.sincos = sinNN(input_size)
        self.hidden_0 = torch.nn.Linear(input_size, hidden_size)

        for _ in range(hidden_layer):
            lys.append(nn.Linear(hidden_size, hidden_size))
            lys.append(nn.ReLU())
        self.ly = nn.Sequential(*lys)
        self.out = nn.Sequential(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        # x = self.sincos(x)
        x = self.hidden_0(x)
        x = self.out(self.ly(x))
        return x

class ik_model(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, output_size):
        super(ik_model, self).__init__()
        lys = []
        lys.append(nn.Linear(input_size, hidden_size))
        lys.append(nn.ReLU())
        for _ in range(hidden_layer):
            lys.append(nn.Linear(hidden_size, hidden_size))
            lys.append(nn.ReLU())
        self.ly = nn.Sequential(*lys)
        self.out = nn.Sequential(nn.Linear(hidden_size, output_size))
    
    def forward(self, x):
        x = self.ly(x)
        x = self.out(x)
        return x


class FK():
    def __init__(self,DH_):
        super().__init__()
        assert DH_.ndim == 2
        assert DH_.shape[0] == 3
        assert DH_[0].shape[0] == DH_[1].shape[0]
        assert DH_[0].shape[0] == DH_[2].shape[0]

        self.alpha = DH_[0][:]
        self.D = DH_[1][:]
        self.L = DH_[2][:]

    def rotate(self, axis, deg):
        AXIS = ('X', 'Y', 'Z')
        axis = str(axis).upper()
        # if axis not in AXIS:
        #     print(f"{axis} is unknown axis, should be one of {AXIS}")
        #     return
        rot_x = axis == 'X'
        rot_y = axis == 'Y'
        rot_z = axis == 'Z'
        rot_mat = np.array([[(np.cos(deg), 1)[rot_x], (0, -np.sin(deg))[rot_z], (0, np.sin(deg))[rot_y], 0],
                            [(0, np.sin(deg))[rot_z], (np.cos(deg), 1)[rot_y], (0, -np.sin(deg))[rot_x], 0],
                            [(0, -np.sin(deg))[rot_y], (0, np.sin(deg))[rot_x], (np.cos(deg), 1)[rot_z], 0],
                            [0, 0, 0, 1]], dtype=np.float32)
        rot_mat = np.where(np.abs(rot_mat) < 1e-10, 0, rot_mat)  # get a small value when np.cos(np.pi/2)
        return rot_mat

    def trans(self, axis, dis):
        AXIS = ('X', 'Y', 'Z')
        axis = str(axis).upper()
        # if axis not in AXIS:
        #     print(f"{axis} is unknown axis, should be one of {AXIS}")
        #     return
        trans_mat = np.eye(4)
        trans_mat[AXIS.index(axis), 3] = dis
        return trans_mat

    def get_DH(self,joints):
        assert len(joints)==len(self.alpha)
        ans = []
        DOF = len(joints)
        for i in range(DOF):
            tmp = [joints[i], self.alpha[i], self.D[i], self.L[i]]
            ans.append(tmp)
        ans = np.array(ans)
        return ans

    def fk(self, joints):
        # thea_1, thea_2, thea_3, thea_4, thea_5, thea_6 = joints
        # DH_pramater: [link, a, d, thea]，注意这里的单位是m
        DH=self.get_DH(joints)
        T = [self.rotate('z', thea_i).dot(self.trans('z', d_i)).dot(self.trans('x', l_i)).dot(self.rotate('x', a_i))
            for thea_i, d_i, l_i, a_i in DH]
        robot = reduce(np.dot, T)
        return  robot


DH_ = [ [-np.pi/2,0,0,np.pi/2],         # alpha
        [0.079,0,0,0],                  # D
        [0.138,0.135,0.147,0.0597]]     # L
DH_ = np.array(DH_)
dobot = FK(DH_)


def load_data(file):
    with open(file,"r") as rf:
        lines = rf.readlines()
        shuffle(lines)
        p = []
        q = []
        for line in lines:
            datas = line.split(" ")
            p.append([float(x)*100 for x in datas[0:3]])
            q.append([float(x)/180 * np.pi for x in datas[3:-1]])
        q = np.array(q)
        p = np.array(p)
    return q,p

def generate_data(data_nums, delta_data = False, q_ = []):
    with open("data.txt","w") as wf:
        q = []
        p = []
        for _ in range(data_nums):
            if not delta_data:
                joint = np.random.rand(4) * np.pi
            else:
                joint = q_ + np.random.rand(4) * np.pi/20
            # joint_ = joint/np.pi * 180
            q.append(joint[0:4])
            # print("joint is ", joint)
            DH_dobot = dobot.fk(joint)
            # print(DH_dobot)
            p.append(DH_dobot[:,-1][0:3] * 100)

            wf.write(str(np.concatenate((joint,DH_dobot[:,-1][0:3]*100))))
            wf.write('\n')
        p = np.array(p)
        q = np.array(q)
    return q,p


def generate_delta_data(q_0,p_0):
        q,p = generate_data(500, True, q_0)
        delta_p = np.array([p_i - p_0 for p_i in p])
        delta_q = np.array([q_i - q_0 for q_i in q])

        delta_p_range = [delta_p.min(0), delta_p.max(0) - delta_p.min(0)]
        delta_q_range = [delta_q.min(0), delta_q.max(0) - delta_q.min(0)]

        # p_0_s = [p_0 for i in range(p.shape[0])]
        # inputs = np.hstack((delta_p,q))
        inputs = np.array(delta_p)
        inputs = np.array(noramlization(inputs))
        outputs = np.array(noramlization(delta_q))
        return inputs, outputs, delta_p_range, delta_q_range

def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    print("min:", minVals,"max",maxVals)
    ranges = maxVals - minVals
    normData = (data - minVals)/ranges
    return normData

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--hidden_size', type=int, help='update steps for finetunning', default=128)
    argparser.add_argument('--hidden_layer', type=int, help='update steps for finetunning', default=3)
    argparser.add_argument('--batch_size', type=int, help='update steps for finetunning', default=40)
    args = argparser.parse_args()

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(224)

    generate_ = True
    get_delta_fk = True
    q_ = []
    p_ = []

    if not get_delta_fk:
        if generate_:
            q,p = generate_data(100)
        else:
            q,p = load_data("real_data.txt")
        #参数设置
        # p = np.array(p)[0:1000]
        # q = np.array(q)[0:1000]
        inputs = q
        outputs = p
    else:
        q_0 = np.array([1.33,-1.25,0.25,0.29])
        DH_dobot = dobot.fk(q_0)
        # print(DH_dobot)
        p_0 = np.array(DH_dobot[:,-1][0:3] * 100)
        inputs, outputs,delta_p_range, delta_q_range = generate_delta_data(q_0, p_0)
        

    test_set = [inputs[int(inputs.shape[0]*0.8):-1], outputs[int(inputs.shape[0]*0.8):-1]]
    inputs = inputs[0:int(inputs.shape[0]*0.8)]
    outputs = outputs[0:int(outputs.shape[0]*0.8)]
    print("inputs.shape, outputs.shape ", inputs.shape, outputs.shape)

    input_size = inputs.shape[1]
    hidden_size = args.hidden_size
    hidden_layer = args.hidden_layer
    output_size = outputs.shape[1]
    batch_size = args.batch_size
    #整体模型架构
    fk_nn = fk_model(input_size, hidden_size, hidden_layer, output_size)

    #损失函数
    cost = torch.nn.MSELoss(reduction='mean')
    #参数
    optimizer_ik = torch.optim.Adam(fk_nn.parameters(), lr = 0.001)
    # 训练网络
    losses = []
    batch_loss = []

    for i in range(50):
        # MINI-Batch方法来进行训练
        # 打印损失
        if i % 10==0:
            losses.append(np.mean(batch_loss))
            test_x = torch.tensor(test_set[0], dtype = torch.float, requires_grad = False)
            test_y = torch.tensor(test_set[1], dtype = torch.float, requires_grad = False)
            prediction_ = fk_nn(test_x)
            test_loss = cost(prediction_, test_y)
            print(i, " training loss ", np.mean(batch_loss), " test loss ", test_loss.data.numpy())
            if get_delta_fk:
                rands = np.random.choice(prediction_.data.numpy().shape[0])
                joint_1 = [delta_q_i * delta_q_range[1] + delta_q_range[0] + q_0 for delta_q_i in prediction_.data.numpy()]
                p_pre = np.array([dobot.fk(joint_i)[:,-1][0:3] * 100 for joint_i in joint_1])
                p_real = [p_0 + inputs_[0:3] * delta_p_range[1] + delta_p_range[0] for inputs_ in test_x.data.numpy()]

                dist_0 = []
                dist_1 = []
                for i in range(0,p_pre.shape[0]):
                    dist_0.append(np.linalg.norm(p_pre[i] - p_real[i]))
                    dist_1.append(np.linalg.norm(p_real[i] - p_0))
                dist_0 = np.mean(np.array(dist_0))
                dist_1 = np.mean(np.array(dist_1))
                print(" mean distance ", dist_0, " distance_2 ", dist_1)

        batch_loss = []
        
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

    if get_delta_fk:
        q_i = []
        p_i = []
        delta_q_i = []
        delta_p_i = []
        for i in range(50):
            # 测试泛化性
            q_i.append(q_0 + np.random.rand(4) * np.pi)
            DH_dobot = dobot.fk(q_i[-1])
            p_i.append(np.array(DH_dobot[:,-1][0:3] * 100))

            delta_q_i.append(np.random.rand(4) * np.pi/10)
            DH_dobot = dobot.fk(q_i[-1] + delta_q_i[-1])
            delta_p_i.append(np.array(DH_dobot[:,-1][0:3] * 100) - p_i[-1])
            delta_q_i[-1] = (delta_q_i[-1] - delta_q_range[0])/delta_q_range[1]
            delta_p_i[-1] = (delta_p_i[-1] - delta_p_range[0])/delta_p_range[1]

        test_x = torch.tensor(delta_p_i, dtype = torch.float, requires_grad = False)
        test_y = torch.tensor(delta_q_i, dtype = torch.float, requires_grad = False)
        print(test_x.shape,test_y.shape)
        prediction_ = fk_nn(test_x)
        test_loss = cost(prediction_, test_y)
        print(" test loss in other points", test_loss.data.numpy())

        rands = np.random.choice(prediction_.data.numpy().shape[0])
        joint_1 = [delta_q_i * delta_q_range[1] + delta_q_range[0] for delta_q_i in prediction_.data.numpy()]
        joint_1 = [joints + delta_joints for joints, delta_joints in zip(q_i, joint_1)]
        p_pre = np.array([dobot.fk(joint_i)[:,-1][0:3] * 100 for joint_i in joint_1])
        p_real = [p_i_s + delta_p_i_s for p_i_s , delta_p_i_s in zip(p_i, delta_p_i)]

        print(p_pre.shape, np.array(p_real).shape)
        dist_0 = []
        dist_1 = []
        for i in range(0,p_pre.shape[0]):
            dist_0.append(np.linalg.norm(p_pre[i] - p_real[i]))
            dist_1.append(np.linalg.norm(p_real[i] - p_i[i]))
        dist_0 = np.mean(np.array(dist_0))
        dist_1 = np.mean(np.array(dist_1))
        print(" mean distance ", dist_0, " distance_2 ", dist_1)
