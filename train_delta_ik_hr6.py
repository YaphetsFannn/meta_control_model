# -*- coding: UTF-8 -*-
"""
    introduction: train a delta inverse kinematics model of hr6
"""
from matplotlib import colors
import  torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from meta import Meta
import  argparse
import numpy as np
from models import ann_model
import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
from fk_models import *
from mpl_toolkits.mplot3d import Axes3D
import train_ik_hr6
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
pku_hr6 = get_Robot()
import time

config = [
            ('linear', [3, 64]),
            ('relu', [True]),
            ('linear', [64, 64]),
            ('relu', [True]),
            ('linear', [64, 6]),
        ]
class testIkM():
    def __init__(self,model_path):
        self.ik_nn = ann_model(train_ik_hr6.config)
        self.ik_nn.load_state_dict(torch.load(model_path))
        # print(self.ik_nn.state_dict())
        # self.publisher = rospy.Publisher()
    def cal_ik(self, positions):
        # print(positions)
        positions = np.array(positions)
        with torch.no_grad():
            test_x = torch.tensor(positions, dtype = torch.float, requires_grad = False)
            prediction_joints = self.ik_nn(test_x)
        return prediction_joints.data.numpy()

def read_min_max(path):
    p_range = []
    q_range = []
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

def get_ik_res(ik_model,position,q_range,p_range):
    inputs = [(position - p_range[0])/p_range[1]]    # shape is [[]]
    print("inputs:",inputs)
    inputs = torch.tensor(inputs, dtype = torch.float, requires_grad = False)
    joints = ik_model(inputs).data.numpy()
    print("outputs:",joints)
    joints = [(joint * q_range[1] + q_range[0]) for joint in joints]
    print("joints:",joints)
    pos_pre = np.array([pku_hr6.cal_fk(joint_i) for joint_i in joints])    
    print("fk reasult:")
    print(pos_pre)
    joint = joints[0]
    return joint

R2D = 180/np.pi

class DIM():
    def __init__(self,config=None,p_start=[],q_start=[],p_tgt=[],batch_size = 20,\
                 epoch = 10, data_nums = 500,\
                 t_vs_v = 0.8,load=False,meta=False,save=False,\
                 ranges=np.pi/20,model_name="test",l2=False):
        if config==None:
           self.config = [
                ('linear', [3, 64]),
                ('relu', [True]),
                ('linear', [64, 64]),
                ('relu', [True]),
                ('linear', [64, 6]),
            ]
        else:
            self.config = config
        # print("config")
        # print(self.config)
        self.fox = 0.1
        self.DeltaModel = ann_model(self.config)
        if meta:
            self.DeltaModel.load_state_dict(torch.load("./model_trained/meta_ik_delta_last.pkl"))
        if load:
            self.DeltaModel.load_state_dict(torch.load("./model_trained/delta_net_1.pkl"))
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
        if l2:
            self.optimizer_ik = torch.optim.Adam(self.DeltaModel.parameters(), lr = 0.005,betas=(0.9,0.999),weight_decay=0.001)
        else:
            self.optimizer_ik = torch.optim.Adam(self.DeltaModel.parameters(), lr = 0.005,betas=(0.9,0.999))
        # self.optimizer_ik = torch.optim.Adam(self.DeltaModel.parameters(), lr = 0.001)
        # 训练网络
        self.losses = []
        self.batch_size = batch_size
        self.epoch = epoch
        self.data_nums = data_nums
        self.t_vs_v = t_vs_v
        self.delta_p_range, self.delta_q_range = read_range_from_file("./data/delta_min_max.txt")
        self.generate_range = ranges
        self.model_name = model_name
    
    def set_q_s(self,q_start):
        self.q_s = q_start
        d2r = np.pi/180
        need_d2r = False
        for q in self.q_s:
            if q > 2*np.pi or q < -2*np.pi:
                need_d2r = True
        if need_d2r:
            self.q_s = np.array([q_i * d2r for q_i in self.q_s])
    
    def generate_data(self):
        self.hr6 = get_Robot()
        self.p_s_cal = self.hr6.cal_fk(self.q_s)
        # notice here
        self.inputs, self.outputs, test_set, _,_ = \
                                    generate_delta_data(self.q_s, self.p_s_cal,\
                                                    self.data_nums,self.t_vs_v,delta_range_ = self.generate_range)
        # print("self.delta_p_range,self.delta_q_range****************************")
        # print(self.delta_p_range,self.delta_q_range)
        # test_set[0] = np.array([np.append(input_,self.p_s) for input_ in test_set[0]])
        self.test_set = [np.array(test_set[0]).astype(float), np.array(test_set[1]).astype(float)]
        # self.inputs = np.array([np.append(input_,self.p_s) for input_ in self.inputs])
        self.inputs = self.inputs.astype(float)
        self.outputs = self.outputs.astype(float)
    
    def train_DIM(self,debug=True):
        torch.manual_seed(222)
        torch.cuda.manual_seed_all(222)
        np.random.seed(222)
        dis_val = []
        loss_val = []
        loss_train = []
        batch_loss = []
        test_x = torch.tensor(self.test_set[0], dtype = torch.float, requires_grad = False)
        test_y = torch.tensor(self.test_set[1], dtype = torch.float, requires_grad = False)
        self.dis_min = []
        self.dis_max = []
        self.dis_mean_test = []
        self.dis_mean_train = []
        delta_q_range = self.delta_q_range
        delta_p_range = self.delta_p_range
        # start_time = time.clock()
        for i in range(self.epoch):
            if i % 10 ==0 and debug:
            # if False:
                with torch.no_grad():
                    # self.losses.append(np.mean(batch_loss))
                    loss_train.append(np.mean(batch_loss))
                    prediction_ = self.DeltaModel(test_x)
                    test_loss = self.cost(prediction_, test_y)
                    loss_val.append(test_loss.data.numpy())
                    self.losses.append(np.mean(test_loss.data.numpy()))

                    joint_pre = [delta_q_i * delta_q_range[1] + delta_q_range[0] +\
                                    self.q_s for delta_q_i in prediction_.data.numpy()]
                    p_pre = np.array([self.hr6.cal_fk(joint_i) for joint_i in joint_pre])
                    p_test = [  self.p_s_cal + inputs_[0:3] * delta_p_range[1] + delta_p_range[0] \
                                for inputs_ in test_x.data.numpy()]
                    if debug:
                        print("deltaq:")
                        print(prediction_.data.numpy()[-1])    
                        print("************************",i,"***********************")
                        print(i, " training loss ", np.mean(batch_loss), \
                                " test loss ", test_loss.data.numpy())                
                        print("joint[i]:")
                        print(joint_pre[-1])
                        print("p_pre")
                        print(p_pre[-1])
                        print("p_real")
                        print(p_test[-1])
                        print("p_cal")
                        print(self.p_s_cal)
                        dist_0 = []
                        dist_1 = []
                        for i in range(0,p_pre.shape[0]):
                            tmp = np.array(p_pre[i] - p_test[i], dtype=np.float64)
                            tmp2 = np.array(self.p_s_cal - p_test[i], dtype=np.float64)
                            dist_0.append(np.linalg.norm(tmp) + self.fox)
                            dist_1.append(np.linalg.norm(tmp2))
                        dist_0 = np.array(dist_0)
                        self.dis_min.append(dist_0.min())
                        self.dis_max.append(dist_0.max())
                        self.dis_mean_test.append(dist_0.mean())
                        dist_0 = np.mean(np.array(dist_0))
                        dist_1 = np.mean(np.array(dist_1))
                        print(" mean distance ", dist_0, " distance_2 ", dist_1)

                        input_train = torch.tensor(self.inputs, dtype = torch.float, requires_grad = True)
                        prediction_ = self.DeltaModel(input_train)
                        joint_pre = [delta_q_i * delta_q_range[1] + delta_q_range[0] +\
                                        self.q_s for delta_q_i in prediction_.data.numpy()]
                        p_pre = np.array([self.hr6.cal_fk(joint_i) for joint_i in joint_pre])
                        p_train = [  self.p_s_cal + inputs_[0:3] * delta_p_range[1] + delta_p_range[0] \
                                    for inputs_ in self.inputs]
                        dist_0 = []
                        for i in range(0,p_pre.shape[0]):
                            tmp = np.array(p_pre[i] - p_train[i], dtype=np.float64)
                            dist_0.append(np.linalg.norm(tmp))
                        dist_0 = np.array(dist_0)
                        self.dis_mean_train.append(dist_0.mean())
                        dist_0 = np.mean(np.array(dist_0))
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
        # end_time = time.clock()
        # print("for the model {}".format(self.model_name))
        # print('Running time: %s Seconds'%(end_time-start_time))
        self.loss_val = loss_val
        self.loss_train = loss_train
        if debug:
            with open("./logs/delta_ik/delta_ik_val"+self.model_name+".txt","w") as wf:
                for loss in loss_val:
                    wf.write(str(loss)+'\n')
            with open("./logs/delta_ik/delta_ik_tran"+self.model_name+".txt","w") as wf:
                for loss in loss_train:
                    wf.write(str(loss)+'\n')
            with open("./logs/delta_ik/delta_ik_dis_test"+self.model_name+".txt","w") as wf:
                for dis in self.dis_mean_test:
                    wf.write(str(dis)+'\n')
            with open("./logs/delta_ik/delta_ik_dis_train"+self.model_name+".txt","w") as wf:
                for dis in self.dis_mean_train:
                    wf.write(str(dis)+'\n')

    def plot_Img(self):
        # plt.legend(loc='lower right', frameon=False) # 图例在图形里面
        # plt.legend(loc=8, frameon=False, bbox_to_anchor=(0.5,-0.3))# 图例在图形外面
        plt.plot(range(len(self.dis_mean_test)),self.dis_mean_test,linestyle=":",label = u"test") 
        plt.plot(range(len(self.dis_mean_test)),self.dis_mean_train,linestyle="-",label = u"training")
        # plt.xlabel(u'迭代次数',fontsize=25)
        # # plt.ylabel('mse loss')
        # plt.ylabel(u"局部反向模型误差(cm)",fontsize=25)
        # plt.title(u'局部反向模型训练过程',fontsize=25)
        plt.fill_between(range(len(self.dis_mean_test)),
                        self.dis_min,
                        self.dis_max,
                        color='b',
                        alpha=0.2,
                        label=u"range")
        plot_point = []
        plt.legend(loc='upper right', fontsize=10)
        plt.show()

        plt.plot(range(len(self.dis_mean_test)),self.loss_val,linestyle=":",label = u"test") 
        plt.plot(range(len(self.dis_mean_test)),self.loss_train,linestyle="-",label = u"training")
        plt.xlabel(u'epoch',fontsize=25)
        # plt.ylabel('mse loss')
        # plt.ylabel(u"局部反向模型均方误差",fontsize=25)
        # plt.title(u'局部反向模型训练过程',fontsize=25)
        # plt.fill_between(range(len(self.dis_mean_test)),
        #                 self.dis_min,
        #                 self.dis_max,
        #                 color='b',
        #                 alpha=0.2,
        #                 label=u"测试集误差区间")
        plot_point = []
        plt.legend(loc='upper right', fontsize=10)
        plt.show()
        # fig = plt.figure()
        # input_ = [np.array((delta_p*self.delta_p_range[1] + self.delta_p_range[0])) for delta_p in self.inputs]
        # input_ = np.array([[d[0]+self.p_s_cal[0],d[1]+self.p_s_cal[1],d[2]+self.p_s_cal[2]] for d in input_])
        # ax = fig.add_subplot(111,projection='3d')
        # ax.scatter(input_[:,0],input_[:,1],input_[:,2],c='b',marker='o')
        # ax.scatter(self.p_s_cal[0],self.p_s_cal[1],self.p_s_cal[2],c='r',marker='^')
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # plt.show()

    def move_delta(self,delta_p):
        input_ = np.array((delta_p-self.delta_p_range[0]) / self.delta_p_range[1])
        # print("delta p :",delta_p)
        input_ = torch.tensor(input_, dtype = torch.float, requires_grad = False)
        # print("input p :",input_.data.numpy())
        
        prediction_ = self.DeltaModel(input_)
        # print("output:",prediction_.data.numpy())
        delta_norme_out = prediction_.data.numpy()
        delta_q_pre = delta_norme_out * self.delta_q_range[1] + self.delta_q_range[0]
        # print("delta q:",delta_q_pre)
        return delta_q_pre

    def go_to_tgt(self):
        delta_p = self.p_t - self.p_s
        return self.move_delta(delta_p)

    def save_model(self):
        path = "./model_trained/delta_net_"+self.model_name+".pkl"
        # torch.save(self.DeltaModel, path)
        torch.save(self.DeltaModel.state_dict(), path)

def str2f(s):
    data = s.split(" ")
    data = [float(d) for d in data]
    return data

def plot_3d_Img(args):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    nums_of_tr = args.p
    theta = np.linspace(-4 * np.pi, 4 * np.pi, nums_of_tr)
    z = np.linspace(-2, 6, nums_of_tr)
    r = z
    x = r * np.sin(theta) + 25
    y = r * np.cos(theta)
    ax.plot(x, y, z,color='b',label=u"目标轨迹")

    ik_real = testIkM("./model_trained/net_param.pkl")
    p_range_real, q_range_real = read_min_max("./model_trained/min_max_generated_data_test.txt")
    # position_tgt = [ 33.68, -14.7283, 6.9515]
    q_cur = get_ik_res(ik_real.ik_nn,[x[0],y[0],z[0]],q_range_real,p_range_real)
    p_cur = pku_hr6.cal_fk(q_cur)
    print("tgt_p,cur p :",[x[0],y[0],z[0]],p_cur,cal_dis([x[0],y[0],z[0]],p_cur))
    position_cur = [[p_cur] for _ in range(4)]
    position_lim = [[] for _ in range(4)]
    position_tgt = []
    nums_of_data = [0.8,0.8,0.2,0.2]
    ranges = [np.pi/20,np.pi/40,np.pi/20,np.pi/40]
    model_names = ["1","2","3","4"]
    colors = ['r','g','r','g']
    lines = ['-','-','--','--']
    deltaModels = []
    for i in range(len(ranges)):
        deltaModel_ = DIM(q_start=q_cur,\
                    p_tgt=position_tgt,\
                    p_start=position_cur[-1],
                    epoch=args.epoch,
                    load=args.l,
                    t_vs_v=nums_of_data[i],
                    range = ranges[i],
                    model_name=model_names[i],
                    data_nums=500)
        deltaModels.append(deltaModel_)
    q_curs = [q_cur for _ in range(4)]
    start = time.clock()
    count = 0
    for p_tgt in zip(x,y,z):
        p_tgt = np.array([p_tgt[0],p_tgt[1],p_tgt[2]])
        position_tgt.append([p_tgt[0],p_tgt[1],p_tgt[2]])
        for i in range(4):
            deltaModels[i].p_s = np.array(position_cur[i][-1])
            deltaModels[i].p_t = np.array([p_tgt[0],p_tgt[1],p_tgt[2]])
            deltaModels[i].q_s = np.array(q_curs[i])
            deltaModels[i].generate_data()
            deltaModels[i].train_DIM()
            delat_q = deltaModels[i].go_to_tgt()
            q_curs[i] = q_curs[i] + delat_q
            p_cur = np.array(pku_hr6.cal_fk(q_curs[i]))
            position_cur[i].append(p_cur)
            position_lim[i].append(p_cur)
        print("now in ",count)
        count = count + 1
    end = time.clock()
    print('Running time: %s Seconds'%(end-start))
    
    # position_cur = []
    # for position_tgt in zip(x,y,z):
    #     delta = [np.random.uniform(low = -0.2, high = 0.2),np.random.uniform(low = -0.2, high = 0.2),np.random.uniform(low = -0.1, high = 0.1)]
    #     position_cur.append(np.array(position_tgt)+delta)
    position_lim = np.array(position_lim)
    position_tgt = np.array(position_tgt)
    position_cur = np.array(position_cur)
    for i in range(4):
        # position_lim[i] = (position_lim[i] + position_tgt)/2
        dis,mean_dis = distance(position_lim[i],position_tgt)
        print("mean_dis of lim_{} is {}".format(i,mean_dis))
        print("loss of lim_{} is {}".format(i,deltaModels[i].losses))
        # print(dis)
        if i == 0:
            ax.plot(position_cur[i,:,0], position_cur[i,:,1], position_cur[i,:,2],\
                    color=colors[i],linestyle=lines[i],label=r"$LIM_1$"+u"末端行进轨迹")
        if i == 1:
            ax.plot(position_cur[i,:,0], position_cur[i,:,1], position_cur[i,:,2],\
                    color=colors[i],linestyle=lines[i],label=r"$LIM_2$"+u"末端行进轨迹")
        if i == 2:
            ax.plot(position_cur[i,:,0], position_cur[i,:,1], position_cur[i,:,2],\
                    color=colors[i],linestyle=lines[i],label=r"$LIM_3$"+u"末端行进轨迹")
        if i == 3:
            ax.plot(position_cur[i,:,0], position_cur[i,:,1], position_cur[i,:,2],\
                    color=colors[i],linestyle=lines[i],label=r"$LIM_4$"+u"末端行进轨迹")
        deltaModels[i].save_model()
        # print("position_lim{} ".format(i),position_lim[i])
    # print("position_tgt ",position_tgt)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()
def main(args):

    deltaModel = DIM(q_start=str2f("108.69 -9.38 51.56 -14.06 -38.09 -16.99"),\
                    p_tgt=str2f("39.4088 -9.5203 4.3103"),\
                    p_start=str2f("38.2714 -8.1530 11.8022"),
                    epoch=args.epoch,
                    data_nums=args.n,
                    load=args.l,
                    meta=args.m,
                    t_vs_v=args.v,
                    l2=args.l2)
    # deltaModel = DIM(q_start=str2f("108.69 -9.38 51.56 -14.06 -38.09 -16.99"))
    deltaModel.generate_data()
    deltaModel.train_DIM()
    deltaModel.plot_Img()
    # plot_3d_Img(args)
    # deltaModel.go_to_tgt()
    deltaModel.save_model()


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
    argparser.add_argument('--l', type=bool, help='load', default=False)
    argparser.add_argument('--m', type=bool, help='load meta', default=False)
    argparser.add_argument('--n', type=int, help='data nums', default=100)
    argparser.add_argument('--p', type=int, help='data nums', default=20)
    argparser.add_argument('--v', type=float, help='training data scale', default=0.5)
    argparser.add_argument('--l2', type=bool, \
                        help='if l2 loss', default=False)
    argparser.add_argument('--generate_data', type=bool, \
                                help='generate radom datas or using true data', default=True)
    args = argparser.parse_args()
    main(args)
