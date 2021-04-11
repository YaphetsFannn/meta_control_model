import numpy as np
import os
import  argparse
from fk_models import *

def main(args):
    input_f = args.i
    datas = []
    with open(os.path.join("./data",input_f+".txt"),'r') as rf:
        lines = rf.readlines()
        for line in lines:
            data = line.strip().split(" ")
            data = [float(data_) for data_ in data]
            datas.append(data)
    pos_real = []
    pos_pre = []
    hr6 = get_Robot()
    for data in datas:
        pos_real.append(data[0:3])
        pos_pre.append(hr6.cal_fk(data[3:]))
    pos_real = np.array(pos_real)
    pos_pre = np.array(pos_pre)
    dis,mean = distance(pos_pre,pos_real)
    print("mean distance is :")
    print(mean)
    # delta_p = [pos_real_ - pos_pre_ for pos_real_ , pos_pre_ in zip(pos_real,pos_pre)]
    delta_p = np.abs(pos_real - pos_pre)
    mean_x = np.mean(delta_p[:,0])
    mean_y = np.mean(delta_p[:,1])
    mean_z = np.mean(delta_p[:,2])
    print(mean_x,mean_y,mean_z)
    max_dis = 3
    dis_delta_p = []
    for idx,p1 in enumerate(pos_real):
        for idx2,p2 in enumerate(pos_real):
            if idx == idx2:
                continue
            dis_real = cal_dis(p1,p2)
            if dis_real > max_dis:
                continue
            delat_p_real = p2 - p1
            delat_p_cal = pos_pre[idx2] - pos_pre[idx]
            dis_delta_p.append(cal_dis(delat_p_cal,delat_p_real))
    dis_delta_p = np.array(dis_delta_p)
    print("shape is ",dis_delta_p.shape)
    print("mean dis:",dis_delta_p.mean())



            
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--i', type=str, help='file name(or path)', default="real_data")
    args = argparser.parse_args()
    
    main(args)