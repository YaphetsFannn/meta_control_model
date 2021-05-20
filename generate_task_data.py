import numpy as np
import os
import  argparse
from fk_models import *
from meta_reg import generate,fk_func,load_joint

def main(args):
    joint_all,joint_all_raw = load_joint()
    task_nums = 10
    for task_count in range(task_nums):
        joint_of_task, pos_of_task = generate(fk_func,joint_all)
        output_f = "task_"+str(task_count)
        with open(os.path.join("./data","ik_tasks",output_f+".txt"),"w") as wf:
            for joint,pos in zip(joint_of_task,pos_of_task):
                for i in range(3):
                    wf.write(str(round(pos[i],2)))
                    wf.write(" ")
                for i in range(6):
                    wf.write(str(round(joint[i],2)))
                    if i != 5:
                        wf.write(" ")
                    else:
                        wf.write("\n")
            print("task ",task_count," done!")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n', type=int, help='number of insert', default=100)
    argparser.add_argument('--i', type=str, help='file name(or path)', default="real_data")
    argparser.add_argument('--o', type=str, help='file name(or path)', default="generated_data")
    args = argparser.parse_args()

    main(args)