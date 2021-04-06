from pubulisher import *
import numpy as np


R2D = 180/np.pi
D2R = np.pi/180
del_ =[0, \
        0, 0, 0, 0,\
        0, 0, 0, 0, \
        -20*D2R, -40*D2R, 20*D2R, -20*D2R,\
        10*D2R, 18*D2R, 0, 0, \
        5*D2R, -15*D2R, 0, 0, \
        0, 0, 0, 0, \
        0, 0, 0, 0]
if __name__ == "__main__":
    Pubs = JointPub()
    angle = [0 for i in range(30)]

    angle[4] = -0.25
    angle[6] = -0.3
    angle[11] = -0.4
    angle[12] = 0.4
    angle[13] = 0.78
    angle[14] = -0.78
    angle[15] = 0.38
    angle[16] = -0.38
    angle[1],angle[3],angle[5],angle[21],angle[23],angle[25] = 1.8205, -0.3268,  0.6991, -0.1311, -0.299 , -0.1978
    # angle[1],angle[3],angle[5],angle[21],angle[23],angle[25] = 90,12,24,25,5,-35
    for  i in range(26):
        data = angle[i] + del_[i];
        if i==0:
            continue
        if(i==1 or i==3 or i==5 or i==21 or i==23 or i==25):
            continue;
        if(i==2 or i==4 or i==6 or i==22 or i==24 or i==26):
            continue;
        Pubs.publish_sigle_joint(data,i)
    for  i in [1,3,5,21,23,25]: # move arm after body
        data = angle[i] + del_[i];
        Pubs.publish_sigle_joint(data,i)