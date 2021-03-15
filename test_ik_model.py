# -*- coding: UTF-8 -*-
"""
    @description: test ik model in real robot
"""
import  torch
from torch import nn
from torch import optim
from torch.nn import functional as F
# from meta import Meta
import  argparse
import numpy as np
from fk_models import *
from models import ann_model
# import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from train_ik_hr6 import *
#define JOINT_NUM 28
from pubulisher import JointPub

R2D = 180/np.pi
class testIkM():
    def __init__(self,model_path):
        self.ik_nn = ann_model(config)
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

if __name__ == "__main__":
    ik_tester = testIkM("./model_trained/net_param.pkl")
    positions = [[31.9740,-11.3775,10.4982],[33.0454,-16.3411,7.3619]]
    positions = np.array(positions)
    p_range = []
    q_range = []
    with open("./model_trained/min_max.txt","r") as rf:
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
    print(p_range)
    print(q_range)
    positions = [position * p_range[1] + p_range[0] for position in positions]
        
    joints = ik_tester.cal_ik(positions)
    joints = [(joint * q_range[1] + q_range[0])*R2D for joint in joints]
    Pubs = JointPub()
    Pubs.publish_jointsD(joints[0])
    print("output: ")
    print(joints)