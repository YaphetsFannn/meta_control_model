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
    def __init__(self,config=None,p_start=[],q_start=[],p_tgt=[],batch_size = 20,\
                 epoch = 10, data_nums = 500,model_path = "./model_trained/delta_net_param.pkl"):
        if config==None:
           self.config = [
                ('linear', [3, 32]),
                ('relu', [True]),
                # ('bn', [128]),

                ('linear', [32, 64]),
                ('relu', [True]),
                # ('bn', [128]),

                ('linear', [64, 128]),
                ('relu', [True]),
                # ('bn', [128]),

                ('linear', [128, 64]),
                ('relu', [True]),
                # ('bn', [128]),

                ('linear', [64, 6]),
                # ('sigmoid', [True])
            ]
        else:
            self.config = config
        print("config")
        print(self.config)
        self.DeltaModel = ann_model(self.config)
        self.DeltaModel.load_state_dict(torch.load(model_path))
        self.q_s = q_start
        d2r = np.pi/180
        need_d2r = False
        for q in self.q_s:
            if q > 2*np.pi or q < -2*np.pi:
                need_d2r = True
        if need_d2r:
            self.q_s = np.array([q_i * d2r for q_i in self.q_s])
        self.p_s = p_start
        self.p_t = p_tgt
        self.cost = torch.nn.MSELoss(reduction='mean')
        #参数
        self.optimizer_ik = torch.optim.Adam(self.DeltaModel.parameters(), lr = 0.001)
        # 训练网络
        self.losses = []
        self.batch_size = batch_size
        self.epoch = epoch
        self.data_nums = data_nums

    def generate_data(self):
        self.hr6 = get_Robot()
        self.p_s_cal = self.hr6.cal_fk(self.q_s)
        self.inputs, self.outputs, test_set, self.delta_p_range, self.delta_q_range = generate_delta_data(self.q_s, self.p_s_cal,self.data_nums,0.1)
        self.test_set = [np.array(test_set[0]).astype(float), np.array(test_set[1]).astype(float)]
        self.inputs = self.inputs.astype(float)
        self.outputs = self.outputs.astype(float)
    
    def train_DIM(self):
        torch.manual_seed(222)
        torch.cuda.manual_seed_all(222)
        np.random.seed(222)
        dis_val = []
        losses_test = []
        batch_loss = []
        test_x = torch.tensor(self.test_set[0], dtype = torch.float, requires_grad = False)
        test_y = torch.tensor(self.test_set[1], dtype = torch.float, requires_grad = False)
        self.dis_min = []
        self.dis_max = []
        self.dis_mean = []
        delta_q_range = self.delta_q_range
        delta_p_range = self.delta_p_range
        for i in range(self.epoch):
            if i % 1==0:
                with torch.no_grad():
                    self.losses.append(np.mean(batch_loss))
                    prediction_ = self.DeltaModel(test_x)
                    test_loss = self.cost(prediction_, test_y)
                    losses_test.append(test_loss.data.numpy())
                    print("************************",i,"***********************")
                    print(i, " training loss ", np.mean(batch_loss), " test loss ", test_loss.data.numpy())

                    joint_1 = [delta_q_i * delta_q_range[1] + delta_q_range[0] +\
                                    self.q_s for delta_q_i in prediction_.data.numpy()]
                    print("joint[i]:")
                    print(joint_1[-1])
                    p_pre = np.array([self.hr6.cal_fk(joint_i) for joint_i in joint_1])
                    print("p_pre")
                    print(p_pre[-1])
                    p_real = [  self.p_s_cal + inputs_[0:3] * delta_p_range[1] + delta_p_range[0] \
                                for inputs_ in test_x.data.numpy()]
                    print("p_real")
                    print(p_real[-1])
                    print(self.p_s_cal)

                    dist_0 = []
                    dist_1 = []
                    for i in range(0,p_pre.shape[0]):
                        tmp = np.array(p_pre[i] - p_real[i], dtype=np.float64)
                        tmp2 = np.array(self.p_s_cal - p_real[i], dtype=np.float64)
                        dist_0.append(np.linalg.norm(tmp))
                        dist_1.append(np.linalg.norm(tmp2))
                    dist_0 = np.array(dist_0)
                    self.dis_min.append(dist_0.min())
                    self.dis_max.append(dist_0.max())
                    self.dis_mean.append(dist_0.mean())
                    dist_0 = np.mean(np.array(dist_0))
                    dist_1 = np.mean(np.array(dist_1))
                    print(" mean distance ", dist_0, " distance_2 ", dist_1)
                        
            batch_loss = []
            # MINI-Batch方法来进行训练
            for start in range(0, len(self.inputs), self.batch_size):
                end = start + self.batch_size if start + self.batch_size < len(self.inputs) else len(self.inputs)
                xx = torch.tensor(self.inputs[start:end], dtype = torch.float, requires_grad = True)
                yy = torch.tensor(self.outputs[start:end], dtype = torch.float, requires_grad = True)
                prediction = self.DeltaModel(xx)
                loss = self.cost(prediction, yy)
                batch_loss.append(loss.data.numpy())
                self.optimizer_ik.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer_ik.step()

    def plot_Img(self):
        # plt.legend(loc='lower right', frameon=False) # 图例在图形里面
        # plt.legend(loc=8, frameon=False, bbox_to_anchor=(0.5,-0.3))# 图例在图形外面
        
        plt.plot(range(len(self.dis_mean)),self.dis_mean) 
        plt.xlabel('epoch')
        # plt.ylabel('mse loss')
        plt.ylabel('distance between p\' and p (cm)')
        plt.fill_between(range(len(self.dis_mean)),
                        self.dis_min,
                        self.dis_max,
                        color='b',
                        alpha=0.2)
        plt.show()
    def save_model(self,path = "./model_trained/delta_net_param.pkl"):
        # torch.save(self.DeltaModel, path)
        torch.save(self.DeltaModel.state_dict(), path)

def str2f(s):
    data = s.split(" ")
    data = [float(d) for d in data]
    return data

def main(args):
    deltaModel = DIM(q_start=str2f("115 -12 60 -14.06 -44 -19"))
    # deltaModel = DIM(q_start=str2f("108.69 -9.38 51.56 -14.06 -38.09 -16.99"))
    deltaModel.generate_data()
    deltaModel.train_DIM()
    deltaModel.plot_Img()
    # deltaModel.save_model()


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