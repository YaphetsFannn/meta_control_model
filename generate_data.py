import numpy as np
import os
import  argparse
from fk_models import *
def main(args):
    input_f = args.i
    output_f = args.o
    joints = []
    with open(os.path.join("./data",input_f+".txt"),'r') as rf:
        lines = rf.readlines()
        data_0 = lines[0].strip().split(" ")
        bias = len(data_0 )- 6
        for line in lines:
            data = line.strip().split(" ")
            data = [float(data_) for data_ in data]
            joints.append(data[bias:])
            
    joints = np.array(joints)
    print("read real data:")
    print(joints.shape)
    v_joints = []
    n = joints.shape[0]
    for i in range(n):
        v_joints.append(joints[i])
        # for _ in range(args.n):
        #     j = np.random.randint(0,n)
        #     v_joints.append([(j_i+j_j)/2 for j_i,j_j in zip(joints[i],joints[j])])
    v_joints = np.array(v_joints)
    print("generate joints:")
    print(v_joints.shape)
    pku_hr6 = get_Robot()
    count = 1
    with open(os.path.join("./data",output_f+".txt"),'w') as wf:
        for joint in v_joints:
            tmp_p = pku_hr6.cal_fk(joint,True)
            if count % 100 == 0:
                print(count)
            count = count + 1
            for i in range(3):
                wf.write(str(round(tmp_p[i],2)))
                wf.write(" ")
            for i in range(6):
                wf.write(str(round(joint[i],2)))
                if i != 5:
                    wf.write(" ")
                else:
                    wf.write("\n")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n', type=int, help='number of insert', default=100)
    argparser.add_argument('--i', type=str, help='file name(or path)', default="real_data")
    argparser.add_argument('--o', type=str, help='file name(or path)', default="generated_data")
    args = argparser.parse_args()

    main(args)