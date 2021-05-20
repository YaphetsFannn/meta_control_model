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
from test_ik_model import read_min_max
FLAGS = argparse.ArgumentParser()
links_len = np.array([-0.69, -20.46,11.68,10.43,10.14,-0.94,-1.68,0.40])
p_range,q_range = read_min_max("./model_trained/min_max.txt")

# Parameters
FLAGS.add_argument('--mode', type=str, choices=['maml', 'reptile'])
FLAGS.add_argument('--n_shot', type=int, default=200,
    help= "How many samples points to regress on while training.")
# FLAGS.add_argument('--train_func', type=str, choices=['sin', 'cos', 'linear'], default='sin',
#     help = "Base function you want to use for traning")
# FLAGS.add_argument('--batch_size', type=int, help='update steps for finetunning', default=40)
FLAGS.add_argument('--iterations', type=int, default=20000)
FLAGS.add_argument('--outer_step_size', type=float, default=0.001)
FLAGS.add_argument('--inner_step_size', type=float, default=0.02)
FLAGS.add_argument('--inner_grad_steps', type=int, default=1)
FLAGS.add_argument('--eval_grad_steps', type=int, default=50)    
FLAGS.add_argument('--eval_iters', type=int, default=5, 
    help='How many testing samples of k different shots you want to run')
FLAGS.add_argument('--logdir', type=str, default="runs", 
    help="TensorBoard Logging")
FLAGS.add_argument('--seed', type=int, default=1)

device = ('cuda' if torch.cuda.is_available() else 'cpu')

prediction_range = 5
# x_all = np.linspace(-prediction_range, prediction_range, num = 50)

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
    joint_all_raw = joint_all_raw * np.pi / 180
    joint_all = np.array([(joint - q_range[0])/q_range[1] for joint in joint_all_raw])
    return joint_all,joint_all_raw

joint_all, joint_all_raw = load_joint()
print("joint shape ",joint_all.shape)

def fk_func(joint_alls,links_len_rand):
    robot = get_Robot_rand(links_len_rand)
    robot_init = get_Robot_rand(links_len)
    random_index = np.random.randint(0, len(joint_alls), 200)
    joint_of_task = joint_alls[random_index]
    positions = np.array([robot.cal_fk(joint) for joint in joint_of_task])
    positions_init = np.array([robot_init.cal_fk(joint) for joint in joint_of_task])
    print("distance between rand and init:",cal_dis(positions[-1],positions_init[-1]))
    positions = np.array([(position_ - p_range[0])/p_range[1] for position_ in positions])
    print("positions.shape ",positions.shape)
    return joint_of_task, positions
    
def generate(func,joint_alls):
    # generate random y
    np.array([-0.69, -20.46,11.68,10.43,10.14,-0.94,-1.68,0.40])
    rand_delta = []
    rand_delta.append(np.random.uniform(low = -0.2, high = 0.2))
    rand_delta.append(np.random.uniform(low = -3.2, high = 3.5))
    rand_delta.append(np.random.uniform(low = -1, high = 1))
    rand_delta.append(np.random.uniform(low = -1, high = 1))
    rand_delta.append(np.random.uniform(low = -1, high = 1))
    rand_delta.append(np.random.uniform(low = -0.5, high = 0.5))
    rand_delta.append(np.random.uniform(low = -1, high = 1))
    rand_delta.append(np.random.uniform(low = -0.5, high = 0.5))
    rand_delta = np.array(rand_delta)
    rand_link = links_len + rand_delta
    print("rand link",rand_link)
    joint_of_task, pos_of_task = func(joint_alls, rand_link)
    return joint_of_task, pos_of_task

def select_points(joints,position, k):
    random_points = np.random.choice(np.arange(len(joints)), k,replace=False)[:, None]
    # print(random_points)
    return joints[random_points], position[random_points] 

def plot_tensorboard(y_eval, pred, k, n , learner, wave_name='SinWave'):
    for j in range(len(y_eval)):
        print('Test_Run_{}/{}/{}/{}_points_sampled'.format(n,
                learner,wave_name,str(k)), 
            'Original Function', y_eval[j], 'Pretrained', pred['pred'][0][j][0], 
                'Gradient_Step_{}'.format(len(pred['pred'])-1), pred['pred'][-1][j][0], j)  

class Meta_Learning:
    def __init__(self, model):
        self.model = model.to(device)
        # self.writer = writer
    
    def train_maml(self, func, k, iterations, outer_step_size, inner_step_size, 
        inner_gradient_steps, tasks=5):
        loss = 0
        batches = 0
        for iteration in range(iterations):
            init_weights = deepcopy(self.model.state_dict())
            # generate new tasks
            joint_of_task, pos_of_task = generate(func,joint_all)
            joint_test,pos_test = select_points(joint_of_task, pos_of_task, k)
            print("joint_test shape",np.array(joint_test).shape)
            meta_params = {}
            for task in range(tasks):
                # sample for meta-update
                joint_train, pos_train = select_points(joint_of_task, pos_of_task, k)
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
                batches += 1
                self.model.load_state_dict(init_weights)
            learning_rate = outer_step_size * (1 - iteration/iterations)
            self.model.load_state_dict({name: init_weights[name] - 
                learning_rate/tasks * meta_params[name] for name in init_weights})
            print('MAML/Training/Loss/', loss/batches, iteration)
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
        out = self.model(x)
        loss = (out - y).pow(2).mean()
        return loss

    def eval(self, pos_all, k, gradient_steps=10, inner_step_size=0.02):
        joint_p,pos_p = select_points(pos_all, k)
        pred = [self.predict(pos_p[:,None])]
        meta_weights = deepcopy(self.model.state_dict())
        for i in range(gradient_steps):
            loss_base = self.train_loss(pos_p,joint_p)
            loss_base.backward()
            for param in self.model.parameters():
                param.data -= inner_step_size * param.grad.data
            pred.append(self.predict(pos_all[:, None]))
        loss = np.power(pred[-1] - y_all,2).mean()
        print("loss is ",loss)
        self.model.load_state_dict(meta_weights)
        return {"pred": pred, "sampled_points":(x_p, y_p)}

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=device)
        return self.model(x).cpu().data.numpy()

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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    k = args.n_shot
    iterations = args.iterations
    funcs = fk_func
    # writer = SummaryWriter(args.logdir)
    # if(args.train_func == 'sin'):
    #     t_f = sin 
    # elif(args.train_func == 'cos'):
    #     t_f = cos
    # else:
    #     t_f = linear

    # model = Meta_Wave(64)
    model = ann_model(train_ik_hr6.config)
    meta = Meta_Learning(model)
    meta.train_maml(funcs, k, iterations, args.outer_step_size, args.inner_step_size,
        args.inner_grad_steps)
    learner = 'maml'


    # eval
    eval_iters = args.eval_iters
    gradient_steps = args.eval_grad_steps
    inner_step_size = 0.01

    pos_eval, _  = generate(f,joint_all)
    for sample in [5,10,20]:
        pred = meta.eval(pos_eval, sample, gradient_steps, inner_step_size)
        # plot_tensorboard(pos_eval, pred, sample, n, learner, wave_name=name)

    # writer.close()

if __name__ == "__main__":
    main()