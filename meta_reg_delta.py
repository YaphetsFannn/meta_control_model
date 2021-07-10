# -*- coding: UTF-8 -*-
from __future__ import division, print_function, absolute_import

import torch 
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter

from models import *
import train_ik_hr6
from fk_models import *
import argparse
import numpy as np 
import matplotlib.pyplot as plt
from copy import deepcopy
import os
# from test_ik_model import read_min_max
import logging
import train_delta_ik_hr6

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

FLAGS = argparse.ArgumentParser()
links_len = np.array([-0.69, -20.46,11.68,10.43,10.14,-0.94,-1.68,0.40])
p_range,q_range = read_min_max("./model_trained/min_max.txt")

# Parameters
FLAGS.add_argument('--mode', type=str, choices=['maml', 'reptile'])
FLAGS.add_argument('--n_shot', type=int, default=20,
    help= "How many samples points to regress on while training.")
# FLAGS.add_argument('--train_func', type=str, choices=['sin', 'cos', 'linear'], default='sin',
#     help = "Base function you want to use for traning")
# FLAGS.add_argument('--batch_size', type=int, help='update steps for finetunning', default=40)
FLAGS.add_argument('--iterations', type=int, default=1000)
FLAGS.add_argument('--outer_step_size', type=float, default=0.005)
FLAGS.add_argument('--inner_step_size', type=float, default=0.01)
FLAGS.add_argument('--inner_grad_steps', type=int, default=20)
FLAGS.add_argument('--eval_grad_steps', type=int, default=30)   
FLAGS.add_argument('--eval_iters', type=int, default=5, 
    help='How many testing samples of k different shots you want to run')
FLAGS.add_argument('--log', type=str, default="./logs/maml_ik_log.txt", 
    help="TensorBoard Logging")
FLAGS.add_argument('--model_name', type=str, default="test")
FLAGS.add_argument('--seed', type=int, default=1)

device = ('cuda' if torch.cuda.is_available() else 'cpu')

prediction_range = 5
# x_all = np.linspace(-prediction_range, prediction_range, num = 50)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("./logs/maml_ik_log.txt",'a',delay=False)
handler.setFormatter(fmt=logging.Formatter("%(message)s"))
logger.addHandler(handler)

def load_joint():
    datas = []
    with open(os.path.join("./data","real_data.txt"),'r') as rf:
            lines = rf.readlines()
            random.shuffle(lines)
            for line in lines:
                data = line.strip().split(" ")
                data = [float(data_) for data_ in data]
                datas.append(data)
    datas = np.array(datas)
    joint_all_raw = datas[:,3:]
    position_all_raw = datas[:,0:3]
    joint_all_raw = joint_all_raw * np.pi / 180
    joint_all = np.array([(joint - q_range[0])/q_range[1] for joint in joint_all_raw])
    return joint_all,joint_all_raw,position_all_raw

joint_all, joint_all_raw, position_all_raw = load_joint()
# print("joint shape ",joint_all.shape)

# def fk_func(joint_alls,links_len_rand,data_nums=600):
#     robot = get_Robot_rand(links_len_rand)
#     robot_init = get_Robot_rand(links_len)
#     random_index = np.random.randint(0, len(joint_alls), data_nums)
#     joint_of_task = joint_alls[random_index]
#     positions = np.array([robot.cal_fk(joint) for joint in joint_of_task])
#     positions_init = np.array([robot_init.cal_fk(joint) for joint in joint_of_task])
#     # print("distance between rand and init:",cal_dis(positions[-1],positions_init[-1]))
#     positions = np.array([(position_ - p_range[0])/p_range[1] for position_ in positions])
#     # print("positions.shape ",positions.shape)
#     return joint_of_task, positions

deltaModel = train_delta_ik_hr6.DIM(q_start=[0,0,0,0,0,0],\
                    p_tgt=[39.4088 ,-9.5203, 4.3103],\
                    p_start=[0,0,0],
                    epoch=10,data_nums = 300,ranges=np.pi/40)
def generate(data_nums_= 300):
    # generate random y
    rand_index = np.random.randint(0,len(joint_all_raw))
    joint_start = joint_all_raw[rand_index]
    p_start = position_all_raw[rand_index]
    deltaModel.q_s = joint_start
    deltaModel.p_s = p_start
    deltaModel.data_nums = data_nums_
    deltaModel.generate_data()
    return deltaModel.outputs, deltaModel.inputs

def select_points(joints,position, k):
    random_points = np.random.choice(np.arange(len(joints)), k,replace=False)[:, None]
    # print(random_points)
    return joints[random_points], position[random_points] 

def plot_tensorboard(y_eval, pred, k, n , learner, wave_name='SinWave'):
    for j in range(len(y_eval)):
        logger.info('Test_Run_{}/{}/{}/{}_points_sampled'.format(n,
                learner,wave_name,str(k)), 
            'Original Function', y_eval[j], 'Pretrained', pred['pred'][0][j][0], 
                'Gradient_Step_{}'.format(len(pred['pred'])-1), pred['pred'][-1][j][0], j)  

class Meta_Learning:
    def __init__(self, model,model_init,model_name="test"):
        self.model = model.to(device)
        self.init_model = model_init.to(device)
        self.train_losses = []
        self.eval_losses = []
        self.model_name = model_name
        # self.writer = writer
    
    def train_maml(self, func, shots, iterations, outer_step_size, inner_step_size, 
        inner_gradient_steps, tasks=5):
        loss = 0
        batches = 0
        for iteration in range(iterations):
            init_weights = deepcopy(self.model.state_dict())
            # logger.info("joint_test shape",np.array(joint_test).shape)
            meta_params = {}
            for task in range(tasks):
                # generate new tasks
                joint_of_task, pos_of_task = generate()
                joint_test,pos_test = select_points(joint_of_task, pos_of_task, shots)
                # sample for meta-update
                joint_train, pos_train = select_points(joint_of_task, pos_of_task, shots)
                for grad_step in range(inner_gradient_steps):
                    loss_base = self.train_loss(pos_train,joint_train)
                    loss_base.backward()
                    for param in self.model.parameters():
                        param.data -= inner_step_size * param.grad.data
                loss_meta = self.train_loss(pos_test, joint_test)
                loss_meta.backward()
                for name,param in self.model.named_parameters():
                    if(task == 0):
                        meta_params[name] =  param.grad.data
                    else:
                        meta_params[name] += param.grad.data
                loss += loss_meta.cpu().data.numpy()
                # loss_meta 表示对于某一个task，计算若干次梯度下降后，网络的误差
                # 我们的目标是学习一个初始参数，使得这一组参数对于所有类似task，在迭代若干次后，平均的loss最小
                # 那么每一次采样若干 task，对每个 task 进行若干次梯度下降
                batches += 1
                self.model.load_state_dict(init_weights)

            learning_rate = outer_step_size * (1 - iteration/iterations)
            self.model.load_state_dict({name: init_weights[name] - 
                learning_rate/tasks * meta_params[name] for name in init_weights})
            if(iteration % 100 == 0):
                logger.info("MAML/Task/test/Loss/ {} {}".format(loss_meta, iteration))
                logger.info("MAML/Training/Loss/ {} {}".format(loss/batches, iteration))
                self.train_losses.append(loss/batches)
        self.write_to("./logs/loss_train_"+str(shots)+"_"+self.model_name+".txt",self.train_losses)
            # if(iteration % 1000 == 0):
            #     pred = self.predict(position_all[:,None])
            #     for i in range(len(position_all)):
            #         print('pretrain_wave_{}'.format(iteration/1000), pred[i][0],i)  

    # def train_reptile(self, func, k, iterations, outer_step_size, inner_step_size, 
    #     inner_gradient_steps):
    #     loss = 0
    #     batches=0
    #     for iteration in range(iterations):
    #         init_weights = deepcopy(self.model.state_dict())
    #         position_all , _ = generate(func,joint_all)
    #         for j in range(inner_gradient_steps):
    #             random_order = np.random.permutation(len(x_all))
    #             for start in range(0,len(x_all), k):
    #                 indicies = random_order[start: start + k][:, None]
    #                 loss_base = self.train_loss(x_all[indicies], y_all[indicies])
    #                 loss_base.backward()
    #                 for param in self.model.parameters():
    #                     param.data -= inner_step_size * param.grad.data
    #                 loss += loss_base.cpu().data.numpy()
    #                 batches += 1
    #         learning_rate = outer_step_size * (1 - iteration/iterations)
    #         curr_weights = self.model.state_dict()
    #         self.model.load_state_dict({name: (init_weights[name] + learning_rate * 
    #             (curr_weights[name] - init_weights[name])) for name in curr_weights})
    #         # self.writer.add_scalar('Reptile/Training/Loss/', loss/batches, iteration)
    #         if(iteration % 1000 == 0):
    #             pred = self.predict(x_all[:,None])
    #             for i in range(len(x_all)):
    #                     print('pretrain_wave_{}'.format(iteration/1000), pred[i][0],i)

    def train_loss(self, x, y):
        x = torch.tensor(x, dtype=torch.float32, device = device)
        y = torch.tensor(y, dtype=torch.float32, device = device)
        self.model.zero_grad()
        # print("x shape ",x.shape)
        out = self.model(x)
        loss = (out - y).pow(2).mean()
        return loss

    def train_loss_cmp(self, x, y):
        x = torch.tensor(x, dtype=torch.float32, device = device)
        y = torch.tensor(y, dtype=torch.float32, device = device)
        self.init_model.zero_grad()
        # print("x shape",x.shape)
        out = self.init_model(x)
        loss = (out - y).pow(2).mean()
        return loss

    def eval(self, joint_eval, pos_eval, k, gradient_steps=40, inner_step_size=0.02):
        joint_p,pos_p = select_points(joint_eval, pos_eval, k)
        pred = [self.predict(pos_p[:,None])]
        meta_weights = deepcopy(self.model.state_dict())
        self.eval_losses=[]
        for i in range(gradient_steps):
            loss_base = self.train_loss(pos_p,joint_p)
            self.eval_losses.append(loss_base.cpu().data.numpy())
            loss_base.backward()
            for param in self.model.parameters():
                param.data -= inner_step_size * param.grad.data
            pred.append(self.predict(pos_eval[:, None]))
        loss = np.power(pred[-1] - joint_eval,2).mean()
        logger.info("eval loss is {}".format(loss))
        self.write_to("./logs/loss_eval_"+str(k)+"_"+self.model_name+".txt",self.eval_losses)
        self.model.load_state_dict(meta_weights)
        # return {"pred": pred, "sampled_points":(x_p, y_p)}

    def eval_comp(self, joint_eval, pos_eval, k, gradient_steps=40, inner_step_size=0.02):
        joint_p,pos_p = select_points(joint_eval, pos_eval, k)
        pred = [self.predict_cmp(pos_p[:,None])]
        init_weights = deepcopy(self.init_model.state_dict())
        self.eval_losses=[]
        for i in range(gradient_steps):
            loss_base = self.train_loss_cmp(pos_p,joint_p)
            self.eval_losses.append(loss_base.cpu().data.numpy())
            loss_base.backward()
            for param in self.init_model.parameters():
                param.data -= inner_step_size * param.grad.data
            pred.append(self.predict_cmp(pos_eval[:, None]))
        loss = np.power(pred[-1] - joint_eval,2).mean()
        logger.info("eval loss cmp is {}".format(loss))
        self.write_to("./logs/loss_eval_cmp_"+str(k)+"_"+self.model_name+".txt",self.eval_losses)
        self.init_model.load_state_dict(init_weights)
        # return {"pred": pred, "sampled_points":(x_p, y_p)}

    def write_to(self, file, list):
        ary = np.array(list)
        with open(file,"w") as wf:
            for a in ary:
                wf.write(str(a)+"\n")

    def save_model(self,path = "./model_trained/meta_ik_delta_40_last.pkl"):
            # torch.save(self.DeltaModel, path)
            torch.save(self.model.state_dict(), path)
            # torch.save(self.model, "./model_trained/meta_ik_delta_net.pkl")

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=device)
        return self.model(x).cpu().data.numpy()

    def predict_cmp(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=device)
        return self.init_model(x).cpu().data.numpy()

class Meta_Wave(nn.Module):
    def __init__(self, units):
        super(Meta_Wave, self).__init__()
        self.inp = nn.Linear(1, units)
        self.layer1 = nn.Linear(units,units)
        self.out = nn.Linear(units, 1)

    def forward(self,x):
        x = torch.tanh(self.inp(x))
        x = torch.tanh(self.layer1(x))
        output = self.out(x)
        return output

def main():
    
    args, unparsed = FLAGS.parse_known_args()
    if len(unparsed) != 0:
        raise NameError("Argument {} not recognized".format(unparsed))
    logger.info("\n Trainnig Args: {}".format(args))
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    shots = args.n_shot
    iterations = args.iterations
    # funcs = fk_func
    # writer = SummaryWriter(args.logdir)
    # if(args.train_func == 'sin'):
    #     t_f = sin 
    # elif(args.train_func == 'cos'):
    #     t_f = cos
    # else:
    #     t_f = linear

    # model = Meta_Wave(64)
    model = ann_model(train_delta_ik_hr6.config)
    model_init = ann_model(train_delta_ik_hr6.config)
    model_name = args.model_name
    meta = Meta_Learning(model,model_init,model_name)
    meta.train_maml(None, shots, iterations, args.outer_step_size, args.inner_step_size,
        args.inner_grad_steps)
    learner = 'maml'
    # eval
    eval_iters = args.eval_iters
    gradient_steps = args.eval_grad_steps
    inner_step_size = 0.005

    joint_eval, pos_eval = generate(1000)
    for sample in [10,50,200]:
        meta.eval(joint_eval, pos_eval, sample, gradient_steps, inner_step_size)
        meta.eval_comp(joint_eval, pos_eval, sample, gradient_steps, inner_step_size)
        # plot_tensorboard(pos_eval, pred, sample, n, learner, wave_name=name)
    meta.save_model()
    # writer.close()

if __name__ == "__main__":
    main()