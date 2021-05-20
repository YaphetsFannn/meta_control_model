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
    inputs = [(position_tgt - p_range[0])/p_range[1]]    # shape is [[]]
    joints = ik_real.cal_ik(inputs)
    print("outputs:",joints)
    joints = [(joint * q_range[1] + q_range[0]) for joint in joints]
    print("joints:",joints)
    pos_pre = np.array([pku_hr6.cal_fk(joint_i) for joint_i in joints])    
    print("fk reasult:")
    print(pos_pre)
    joint = joints[0]
    return joint

if __name__ == "__main__":
    need_delta = False
    Pubs = JointPub()    
    ik_real = testIkM("./model_trained/gene_2_net_param.pkl")
    p_range_real, q_range_real = read_min_max("./model_trained/gene_2_min_max.txt")

    pku_hr6 = get_Robot()
    while True:
        msg = rospy.wait_for_message('/hand_position',position,timeout=10)
        position_tgt = np.array([msg.ox - 0.8,msg.oy,msg.oz])
        while position_tgt[0]<5:
            msg = rospy.wait_for_message('/hand_position',position,timeout=10)
            position_tgt = np.array([msg.ox - 0.8,msg.oy,msg.oz])
        print("get msg:")
        print(msg)
        print("position_tgt:")
        print(position_tgt)
        joint = get_ik_res(ik_real,position_tgt,q_range_real,p_range_real)

        Pubs.publish_jointsR(joint)
        rospy.sleep(3)
        msg = rospy.wait_for_message('/hand_position',realsense_camera.msg.position,timeout=10)
        pos_hand = np.array([msg.hx,msg.hy,msg.hz])
        delta_move_count = 0
        while cal_dis(pos_hand,position_tgt) > 1.0:
            while pos_hand[0]<5 or position_tgt[0]<5:
                msg = rospy.wait_for_message('/hand_position',position,timeout=10)
                pos_hand = np.array([msg.hx,msg.hy,msg.hz])
                position_tgt = np.array([msg.ox,msg.oy,msg.oz])
            print("*********************",delta_move_count,"**************************")
            print("pos_hand is:",pos_hand)
            print("pos_tgt is:",position_tgt)
            msg_joint =rospy.wait_for_message('/motor_states/pan_tilt_port',MotorStateList,timeout=10)
            joints_start,valid = get_joint_angle(msg_joint)
            print("joint_start:",joints_start)
            while not valid:
                print("not valid joint")
                msg_joint =rospy.wait_for_message('/motor_states/pan_tilt_port',MotorStateList,timeout=10)
                joints_start,valid = get_joint_angle(msg_joint)
            deltaModel = DIM(q_start=joints_start,\
                        p_tgt=position_tgt,\
                        p_start=pos_hand,\
                        epoch=10,\
                        data_nums=300,
                        t_vs_v=0.9)
            deltaModel.generate_data()
            deltaModel.train_DIM(debug = False)
            # deltaModel.plot_Img()
            deltaModel.save_model()
            delta_q = deltaModel.go_to_tgt()
            delta_q = [d_r * R2D for d_r in delta_q]
            # print("delta_q D:",delta_q)
            # print("joint_start:",joints_start)
            joint_aft_delta = delta_q + joints_start
            print("pos_hand_cal is:",pku_hr6.cal_fk(joint_aft_delta))
            Pubs.publish_jointsD(joint_aft_delta)
            rospy.sleep(3)
            delta_move_count  = delta_move_count + 1
            
            msg = rospy.wait_for_message('/hand_position',position,timeout=100)
            pos_hand = np.array([msg.hx,msg.hy,msg.hz])
            position_tgt = np.array([msg.ox,msg.oy,msg.oz])
            while pos_hand[0]<5 or position_tgt[0]<5:
                msg = rospy.wait_for_message('/hand_position',position,timeout=10)
                pos_hand = np.array([msg.hx,msg.hy,msg.hz])
                position_tgt = np.array([msg.ox,msg.oy,msg.oz])
            print("pos_hand is:",pos_hand)
            print("pos_tgt is:",position_tgt)
            print("distante:",cal_dis(pos_hand,position_tgt))
            print("***********************************************")
            
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