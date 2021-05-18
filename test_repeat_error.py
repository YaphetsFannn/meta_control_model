# descriprion:
#   author: yuanyifan@pku.edu.cn
#   test repeat error. attention, need "rosrun realsense_camera get_hand_position" first
import rospy
from pubulisher import JointPub
import numpy as np
from std_msgs.msg import Header
from realsense_camera.msg import position
import realsense_camera.msg
from train_delta_ik_hr6 import DIM
from save_datas import get_joint_angle
from dynamixel_msgs.msg import MotorStateList
from fk_models import *

def main():
    Pubs = JointPub()
    joint1 = [90,22,24,25,5,-35]
    joint2 = [37.79,12.60,61.52,72.66,-12.01,-54.20]
    joint3 = [60.94,0.29,62.99,34.86,-41.31,-26.37]
    joint4 = [85.84,16.11,48.93,52.44,-2.93,-4.98]
    joint5 = [43.65,70.90,53.61,72.66,5.27,-85.84]
    joint = [joint1,joint2,joint3,joint4,joint5]
    nums = len(joint)
    pos = [[] for i in range(nums)]
    for _ in range(20):
        for i in range(nums):
            Pubs.publish_jointsD(joint[i])
            rospy.sleep(3)
            msg = rospy.wait_for_message('/hand_position',position,timeout=10)
            pos_hand = np.array([msg.hx,msg.hy,msg.hz])
            while pos_hand[0]<5 :
                msg = rospy.wait_for_message('/hand_position',position,timeout=10)
                pos_hand = np.array([msg.hx,msg.hy,msg.hz])
            pos[i].append(pos_hand)
        print(pos[i])
        print("-------------------")
    pos = np.array(pos)
    all_dis = []
    all_arix_dis = [[],[],[]]
    for i in range(nums):
        mean_pos = []
        mean_dis = []
        print("________________________________")
        for j in range(3):
            pos_ = np.array(pos[i][:,j])
            print(pos_)
            print("mean\tvar\tstd:")
            print(np.mean(pos_),np.var(pos_),np.std(pos_))
            mean_pos.append(np.mean(pos_))
            all_arix_dis[j].append(np.var(pos_))
        for pos_hand in pos[i]:
            dis_ = cal_dis(mean_pos,pos_hand)
            mean_dis.append(dis_)
        mean_dis = np.array(mean_dis)
        print("mean dis:",np.mean(mean_dis))
        all_dis.append(np.mean(mean_dis))
    all_dis = np.array(all_dis)
    print("mean dis over all:",np.mean(all_dis))
    all_arix_dis = np.array(all_arix_dis)
    print("mean dis in x|y|z:")
    print(np.mean(all_arix_dis[0]),np.mean(all_arix_dis[1]),np.mean(all_arix_dis[2]))

if __name__ == '__main__':
    main()