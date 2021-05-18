import numpy as np
import os
import  argparse
from fk_models import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def cal_angle(v1_,v2_):
    # print("angle:",v1_,v2_)
    v1 = np.array(v1_)
    v2 = np.array(v2_)
    Lx = np.sqrt(v1.dot(v1))
    Ly = np.sqrt(v2.dot(v2))
    cos_angle = v1.dot(v2)/(Lx*Ly)
    angle = np.arccos(cos_angle)
    return angle/pi * 180

def main(args):
    input_f = args.i
    datas = []
    with open(os.path.join("./data",input_f+".txt"),'r') as rf:
        lines = rf.readlines()
        random.shuffle(lines)
        for line in lines:
            data = line.strip().split(" ")
            data = [float(data_) for data_ in data]
            datas.append(data)
    pos_real = []
    pos_pre = []
    angle_ = []
    hr6 = get_Robot()
    for data in datas:
        pos_i = data[0:3]
        pos_real.append(pos_i)
        # pos_i_fake = [p*random.uniform(0.98,1.02) for p in pos_i]
        # pos_pre.append(pos_i_fake)
        pos_pre_i = hr6.cal_fk(data[3:])
        pos_pre.append(pos_pre_i)
        angle_.append(cal_angle(pos_pre_i,pos_i))
    pos_real = np.array(pos_real)
    pos_pre = np.array(pos_pre)
    dis,mean = distance(pos_pre,pos_real)
    angle_ = np.array(angle_)
    print("mean distance is :")
    print(mean)
    print("mean angle is:")
    print(angle_.mean())
    # delta_p = [pos_real_ - pos_pre_ for pos_real_ , pos_pre_ in zip(pos_real,pos_pre)]
    delta_p = np.abs(pos_real - pos_pre)
    mean_x = np.mean(delta_p[:,0])
    mean_y = np.mean(delta_p[:,1])
    mean_z = np.mean(delta_p[:,2])
    print(mean_x,mean_y,mean_z)
    
    # nums_plt = 20
    # xr,yr,zr = pos_real[0:nums_plt,0],pos_real[0:nums_plt,1],pos_real[0:nums_plt,2]
    # xb,yb,zb = pos_pre[0:nums_plt,0],pos_pre[0:nums_plt,1],pos_pre[0:nums_plt,2]

    # fig = plt.figure()
    # ax = fig.add_subplot(111,projection='3d')
    # ax.scatter(xr,yr,zr,c='r',marker='o')
    # ax.scatter(xb,yb,zb,c='b',marker='^')
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # plt.show()
    
    # test delta distance
    max_dis = 3
    dis_delta_p = []
    angle_delta_p = []
    for idx,p1 in enumerate(pos_real):
        for idx2,p2 in enumerate(pos_real):
            if idx == idx2:
                continue
            dis_real = cal_dis(p1,p2)
            if dis_real > max_dis:
                continue
            delat_p_real = p2 - p1
            delat_p_cal = pos_pre[idx2] - pos_pre[idx]
            # print(delat_p_cal,delat_p_real,cal_angle(delat_p_cal,delat_p_real))
            # print(cal_dis(delat_p_cal,delat_p_real))
            dis_delta_p.append(cal_dis(delat_p_cal,delat_p_real))
            angle_delta_p.append(cal_angle(delat_p_cal,delat_p_real))
    dis_delta_p = np.array(dis_delta_p)
    angle_delta_p = np.array(angle_delta_p)
    print("shape is ",dis_delta_p.shape,angle_delta_p.shape)
    print("mean dis:",dis_delta_p.mean())
    print("mean angle:",np.nanmean(angle_delta_p))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--i', type=str, help='file name(or path)', default="real_data")
    args = argparser.parse_args()
    
    main(args)