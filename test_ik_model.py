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
import rospy
from std_msgs.msg import Header
from realsense_camera.msg import position

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

    pku_hr6 = get_Robot()
    position_tgt = [[28.4599,-2.9071,16.9371],[28.5224,-3.5886,18.9614]]
    #                                        32.7593 , 2.5641, -0.6474
    print("input:")
    print(position_tgt)
    position_tgt = np.array(position_tgt)
    inputs = [(p - p_range[0])/p_range[1] for p in position_tgt]
    print(inputs)
    joints = ik_tester.cal_ik(inputs)
    print("outputs:",joints)
    joints = [(joint * q_range[1] + q_range[0]) for joint in joints]
    print("joints:",joints)
    pos = [pku_hr6.cal_fk(joint_i)[:,-1][0:3] for joint_i in joints]
    #!!! notice that p_real[x,y,z] = p_fk[-y,-z,x]
    pos_pre = np.array([ [-p[1],-p[2],p[0]] for p in pos])
    print("fk reasult:")
    print(pos_pre)
    pos_real = []
    for joint in joints:
        Pubs = JointPub()
        Pubs.publish_jointsR(joint)
        rospy.sleep(3)
        msg = rospy.wait_for_message('/hand_position',position,timeout=100)
        pos_real.append([msg.A,msg.B,msg.C])
    pos_real = np.array(pos_real)
    print("real pos:")
    print(pos_real)
    dis,mean = distance(pos_pre,pos_real)

    print("distance btw fk and real:")
    print(dis)
    print(mean)
    print("distance btw ik and real:")
    dis,mean = distance(pos_real,position_tgt)    
    print(dis)
    print(mean)

    # print(msg.A,msg.B,msg.C)