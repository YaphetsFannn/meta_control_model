# -*- coding: UTF-8 -*-
"""
    @description: 
        get datas from radomly position
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import  argparse
import os
import sys
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from std_msgs.msg import Float64
from dynamixel_msgs.msg import MotorStateList
from pubulisher import JointPub
from realsense_camera.msg import position
import realsense_camera.msg

def get_joint_angle(msgs):
    motor_states = msgs.motor_states
    joints = []
    joints_ids = [1,3,5,21,23,25]
    joints_counts = []
    zero_position = {1:512, 3:506, 5:502, 21:563, 23:536, 25:578}
    step_angle = 300.0/1024
    for joints_id in joints_ids:
        for motor in motor_states:
            if motor.id == joints_id:
                angle = (motor.position - zero_position[joints_id]) * step_angle
                joints.append(angle)
                joints_counts.append(motor.id)
                break
    valid = (len(joints)==6)
    if not valid:
        print(joints_counts)
        
    return np.array(joints),valid


def main(args):

    if args.t: # read tracking datas
        Pubs = JointPub()
        tracking_datas = []
        counts = 0
        with open("/home/pku-hr6/yyf_ws/data/arm_running_300.txt",'r') as rf:
            lines = rf.readlines()
            sub_range = len(lines)//50
            for line in lines:
                datas = line.strip().split(" ")
                print(datas)
                datas = [float(data) for data in datas]
                print(datas)
                if counts % sub_range == 0:
                    tracking_datas.append(datas)
    else:
        rospy.init_node('data_saver')
        

    data_nums = args.n
    file_names = args.file
    datas = []
    save_files = os.path.join("./data",file_names+'.txt')
    count = 0
    with open(save_files,"w") as wf:
        # while count < data_nums:
        while len(datas) < data_nums:
        # while True:
            # Wait for a coherent pair of frames: depth and color
            if args.t:
                if count>=len(tracking_datas) - 1:
                    break
                tracking_data = tracking_datas[count]
                Pubs.publish_jointsD(tracking_data)
                count = count + 1
            depth_msg = rospy.wait_for_message('/hand_position',position,timeout=10)
            depth_point = np.array([depth_msg.hx,depth_msg.hy,depth_msg.hz])
            wait_count = 0
            while depth_point[0]<5 :
                depth_msg = rospy.wait_for_message('/hand_position',position,timeout=10)
                depth_point = np.array([depth_msg.hx,depth_msg.hy,depth_msg.hz])
                wait_count += 1
                if wait_count > 10:
                    break
            if wait_count > 10:
                count += 1
                continue
            print(depth_point)

            msg = rospy.wait_for_message('/motor_states/pan_tilt_port',MotorStateList,timeout=10)
            # print(len(msg.motor_states))
            joints,valid_q = get_joint_angle(msg)
            if valid_q:
                depth_point = np.array(depth_point)
                # print(joints)
                # print(depth_point)
                if args.only_q:
                    data = joints
                else:   
                    data = np.append(depth_point, joints)
                print(data)
                datas.append(data)
            if cv2.waitKey(100) & 0xFF == ord('q'):   # quit
                break
        for data in datas:
            for i in range(len(data)):
                wf.write(str(round(data[i],2)))
                if i != len(data) - 1:
                    wf.write(" ")
                else:
                    wf.write("\n")




if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n', type=int, help='number of datas', default=300)
    argparser.add_argument('--file', type=str, help='file name(or path)', default="q_p_tests")
    argparser.add_argument('--only_q', type=bool, help='type of save datas, only q or p&q', default=False)
    argparser.add_argument('--t', type=bool, help='read track from file', default=True)
    import time
    start = time.time()
    args = argparser.parse_args()
    main(args)
    end = time.time()
    print("times using:{}".format(end-start))