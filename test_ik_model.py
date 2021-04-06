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
import realsense_camera.msg
from train_delta_ik_hr6 import DIM
from save_datas import get_joint_angle
from dynamixel_msgs.msg import MotorStateList

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
    Pubs = JointPub()    
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
    # position_tgt = [[28.4599,-2.9071,16.9371],[28.5224,-3.5886,18.9614]]
    msg = rospy.wait_for_message('/hand_position',position,timeout=10)
    position_tgt = np.array([[msg.ox - 0.8,msg.oy,msg.oz]])
    while position_tgt[0][0]<5:
        msg = rospy.wait_for_message('/hand_position',position,timeout=10)
        position_tgt = np.array([[msg.ox,msg.oy,msg.oz]])
    inputs = [(p - p_range[0])/p_range[1] for p in position_tgt]
    
    print("get msg:")
    print(msg)
    print("position_tgt:")
    print(position_tgt)
    print(inputs)

    joints = ik_tester.cal_ik(inputs)    
    print("outputs:",joints)
    joints = [(joint * q_range[1] + q_range[0]) for joint in joints]
    print("joints:",joints)
    pos_pre = np.array([pku_hr6.cal_fk(joint_i) for joint_i in joints])    
    print("fk reasult:")
    print(pos_pre)
    pos_real = []
    for joint in joints:
        Pubs.publish_jointsR(joint)
        rospy.sleep(3)
        msg = rospy.wait_for_message('/hand_position',realsense_camera.msg.position,timeout=10)
        pos_hand = np.array([msg.hx,msg.hy,msg.hz])
        while pos_hand[0]<5:
            msg = rospy.wait_for_message('/hand_position',position,timeout=10)
            pos_hand = np.array([msg.hx,msg.hy,msg.hz])
        print("pos_hand is:",pos_hand)
        print("pos_tgt is:",position_tgt)
        msg_joint =rospy.wait_for_message('/motor_states/pan_tilt_port',MotorStateList,timeout=10)
        joints_start,valid = get_joint_angle(msg_joint)
        print("joint_start:",joints_start)
        if not valid:
            print("not valid joint")
            break
        deltaModel = DIM(q_start=joints_start,\
                    p_tgt=position_tgt,\
                    p_start=pos_hand)
        deltaModel.generate_data()
        deltaModel.train_DIM(False)
        # deltaModel.plot_Img()
        delta_q = deltaModel.go_to_tgt()[0]
        delta_q = [d_r * R2D for d_r in delta_q]
        print("delta_q:",delta_q)
        print("joint_start:",joints_start)
        joint_aft_delta = delta_q + joints_start
        Pubs.publish_jointsD(joint_aft_delta)
        rospy.sleep(3)
        msg = rospy.wait_for_message('/hand_position',position,timeout=100)
        pos_hand = np.array([msg.hx,msg.hy,msg.hz])
        print("pos_hand is:",pos_hand)
        print("pos_tgt is:",position_tgt)
    # deltaModel.go_to_tgt()
    # deltaModel.save_model()`
        # pos_real.append()
    # pos_real = np.array(pos_real)
    # print("real pos:")
    # print(pos_real)
    # dis,mean = distance(pos_pre,pos_real)

    # print("distance btw fk and real:")
    # print(dis)
    # print(mean)
    # print("distance btw ik and real:")
    # dis,mean = distance(pos_real,position_tgt)    
    # print(dis)
    # print(mean)

    # print(msg.A,msg.B,msg.C)