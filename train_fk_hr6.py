# -*- coding: UTF-8 -*-
"""
    introduction: train a fk model of hr6
"""
import  torch
from torch import nn
from torch import optim
from torch.nn import functional as F
# from meta import Meta
import  argparse
import numpy as np
from fk_models import *
from models import *
# import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

#整体模型架构
config = [
    ('linear', [6, 64]),
    ('relu', [True]),
    # ('bn', [64]),
    ('linear', [64, 64]),
    ('relu', [True]),
    # # ('bn', [128]),

    # # # ('linear', [256, 128]),
    # # # ('relu', [True]),
    # # # # ('bn', [128]),

    # ('linear', [64, 64]),
    # ('relu', [True]),
    # # ('bn', [32]),
    # ('linear', [32, 32]),
    # ('relu', [True]),
    ('linear', [64, 3]),
    # ('sigmoid', [True])
]
input_range, output_range = [],[]
def main(args):
    global config,input_range, output_range
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    print(args)
        #整体模型架构
    generate_ = args.g
    is_delta_model = args.is_delta_model

    if not is_delta_model:
        if generate_:
            inputs, outputs, test_inputs, test_outputs, input_range, output_range = generate_data(1000, is_fk=args.is_fk)
        else:
            # inputs, outputs, test_inputs, test_outputs, input_range, output_range = load_data("./data/"+args.file+".txt", is_fk=args.is_fk)
            outputs, inputs, test_outputs,test_inputs, output_range, input_range = load_data("./data/"+args.file+".txt", is_fk=args.is_fk)
    else:
        q_0 = np.array([1.33,1.02,0.25,0.29,0.3,0.2])
        pku_hr6 = get_Robot()
        p_0 = pku_hr6.cal_fk(q_0)
        inputs, outputs, test_set, delta_input_range, delta_output_range = generate_delta_data(q_0, p_0)
        test_inputs = np.array(test_set[0])
        test_outputs = np.array(test_set[1])
    test_set = [test_inputs.astype(float), test_outputs.astype(float)]
    inputs = inputs.astype(float)
    outputs = outputs.astype(float)
    print(test_set[0][0],test_set[1].shape)
    print("inputs.shape, outputs.shape ", inputs.shape, outputs.shape)
    print("inputs[0]: ", inputs[0])
    print("outputs[0]: ", outputs[0])
    print("input_range: ", input_range)
    print("output_range: ", output_range)
    input_size = inputs.shape[1]
    hidden_size = args.hidden_size
    hidden_layer = args.hidden_layer
    output_size = outputs.shape[1]

    fk_nn = ann_model(config)

    #损失函数
    cost = torch.nn.MSELoss(reduction='mean')
    #参数
    optimizer_fk = torch.optim.Adam(fk_nn.parameters(), lr = 0.0005,betas=(0.9,0.999))
    # 训练网络
    losses_train = []
    batch_size = args.batch_size
    pku_hr6 = get_Robot()
    dis_train = []
    dis_test = []
    losses_test = []
    batch_loss = []
    for i in range(args.epoch):
        if i % 1==0:
            # for test
            with torch.no_grad():
                losses_train.append(np.mean(batch_loss))
                test_x = torch.tensor(test_set[0], dtype = torch.float, requires_grad = False)
                test_y = torch.tensor(test_set[1], dtype = torch.float, requires_grad = False)
                prediction_ = fk_nn(test_x)
                test_loss = cost(prediction_, test_y)
                losses_test.append(test_loss.data.numpy())
                print("************************* epoch ",i,"*************************")
                print(i, " training loss ", np.mean(batch_loss), " test loss ", test_loss.data.numpy())
                if is_delta_model:
                    rands = np.random.choice(prediction_.data.numpy().shape[0])
                    joint_1 = [delta_q_i * delta_output_range[1] + delta_output_range[0] +\
                                 q_0 for delta_q_i in prediction_.data.numpy()]
                    p_pre = np.array([pku_hr6.cal_fk(joint_i) for joint_i in joint_1])
                    p_real = [  p_0 + inputs_[0:3] * delta_input_range[1] + delta_input_range[0] \
                                for inputs_ in test_x.data.numpy()]

                    dist_0 = []
                    dist_1 = []
                    for i in range(0,p_pre.shape[0]):
                        tmp = np.array(p_pre[i] - p_real[i], dtype=np.float64)
                        tmp2 = np.array(p_0 - p_real[i], dtype=np.float64)
                        dist_0.append(np.linalg.norm(tmp))
                        dist_1.append(np.linalg.norm(tmp2))
                        # print(dist_1[-1])
                    joint_1 = [delta_q_i * delta_output_range[1] + delta_output_range[0] +\
                                 q_0 for delta_q_i in prediction_.data.numpy()]
                    dis_train.append(dist_0)
                    dist_0 = np.mean(np.array(dist_0))
                    dist_1 = np.mean(np.array(dist_1))
                    print(" mean distance ", dist_0, " distance_2 ", dist_1)
                else:
                    if args.d:
                        # fk models
                        test_x = torch.tensor(test_set[0], dtype = torch.float, requires_grad = False)
                        norm_inputs = test_x.data.numpy()
                        inputs_tst = np.array([ii*input_range[1] + input_range[0] for ii in norm_inputs])
                        prediction_ = fk_nn(test_x)
                        dist_0 = []
                        dist_1 = []
                        p_pre_fk = np.array([pku_hr6.cal_fk(joint_i) for joint_i in inputs_tst])
                        p_pre_nn = np.array([pp * output_range[1] + output_range[0] for pp in prediction_.data.numpy()])
                        p_real = np.array([pp * output_range[1] + output_range[0] for pp in test_y.data.numpy()])
                        print("p_pre_fk|p_pre_nn|p_real|")
                        print(p_pre_fk[-1],p_pre_nn[-1],p_real[-1])
                        print(prediction_.data.numpy()[-1],test_y.data.numpy()[-1])
                        for i in range(0,p_real.shape[0]):
                            tmp = np.array(p_real[i] - p_pre_fk[i], dtype=np.float64)
                            tmp2 = np.array(p_real[i] - p_pre_nn[i], dtype=np.float64)
                            dist_0.append(np.linalg.norm(tmp))
                            dist_1.append(np.linalg.norm(tmp2))
                        dis_test.append(dist_1)
                        dist_0 = np.mean(np.array(dist_0))
                        dist_1 = np.mean(np.array(dist_1))
                        print(" mean distance ", dist_0, " distance_2 ", dist_1)
                        
                    
        batch_loss = []
        # MINI-Batch方法来进行训练
        for start in range(0, len(inputs), batch_size):
            end = start + batch_size if start + batch_size < len(inputs) else len(inputs)
            rand_index = np.random.randint(0, len(inputs), batch_size)
            xx = torch.tensor(inputs[rand_index], dtype = torch.float, requires_grad = True)
            yy = torch.tensor(outputs[rand_index], dtype = torch.float, requires_grad = True)
            # xx = torch.tensor(inputs[start:end], dtype = torch.float, requires_grad = True)
            # yy = torch.tensor(outputs[start:end], dtype = torch.float, requires_grad = True)
            prediction = fk_nn(xx)
            loss = cost(prediction, yy)
            batch_loss.append(loss.data.numpy())
            optimizer_fk.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_fk.step()
            if start == 200:
                with torch.no_grad():
                    print("\n--------------------------------train------------------------------")
                    norm_inputs = xx.data.numpy()
                    inputs_tr = np.array([ii*input_range[1] + input_range[0] for ii in norm_inputs])
                    p_pre_fk = np.array([pku_hr6.cal_fk(joint_i) for joint_i in inputs_tr])
                    p_pre_nn = np.array([pp * output_range[1] + output_range[0] for pp in prediction.data.numpy()])
                    p_real = np.array([pp * output_range[1] + output_range[0] for pp in yy.data.numpy()])
                    print("p_pre_fk|p_pre_nn|p_real|")
                    print(p_pre_fk[-1])
                    print(p_pre_nn[-1])
                    print(p_real[-1])
                    dist_0 = []
                    dist_1 = []
                    for i in range(0,p_real.shape[0]):
                        tmp = np.array(p_real[i] - p_pre_fk[i], dtype=np.float64)
                        tmp2 = np.array(p_real[i] - p_pre_nn[i], dtype=np.float64)
                        dist_0.append(np.linalg.norm(tmp))
                        dist_1.append(np.linalg.norm(tmp2))
                    dis_train.append(dist_1)
                    dist_0 = np.mean(np.array(dist_0))
                    dist_1 = np.mean(np.array(dist_1))
                    print(" mean distance ", dist_0, " distance_2 ", dist_1)
                    # print(prediction.data.numpy()[-1],yy.data.numpy()[-1])
                    print("--------------------------------train------------------------------\n")
        # dis_train.append(batch_loss)
    with open("./data/fk_loss_val.txt","w") as wf:
        for loss in losses_test:
            wf.write(str(loss)+'\n')
    with open("./data/fk_loss_tran.txt","w") as wf:
        for loss in losses_train:
            wf.write(str(loss)+'\n')
    if args.d:
        dis_min = []
        dis_max = []
        dis_mean = []
        for i in range(len(dis_train)):
            loss_i = np.array(dis_train[i])
            # dis_min.append(loss_i.min())
            # dis_max.append(loss_i.max())
            dis_min.append(np.percentile(loss_i,25))
            dis_max.append(np.percentile(loss_i,75))
            dis_mean.append(loss_i.mean())
        with open("./data/fk_dis.txt","w") as wf:
            for d_min,d_max,d_mean in zip(dis_min,dis_max,dis_mean):
                wf.write(str(d_min)+","+str(d_max)+","+str(d_mean)+'\n')
        # plt.legend(loc='lower right', frameon=False) # 图例在图形里面
        # plt.legend(loc=8, frameon=False, bbox_to_anchor=(0.5,-0.3))# 图例在图形外面
        print("mean dis:")
        print(dis_mean[-1])
        plt.fill_between(range(len(dis_train)),
                dis_min,
                dis_max,
                color='b',
                alpha=0.2)
        plt.plot(range(len(dis_train)),dis_mean,color='g')
        dis_mean = []
        for i in range(len(dis_test)):
            loss_i = np.array(dis_test[i])
            # dis_min.append(loss_i.min())
            # dis_max.append(loss_i.max())
            # dis_min.append(np.percentile(loss_i,25))
            # dis_max.append(np.percentile(loss_i,75))
            dis_mean.append(loss_i.mean())
        plt.plot(range(len(dis_test)),dis_mean,color='r')
        plt.xlabel('epoch')
        # plt.ylabel('mse loss')
        plt.ylabel('distance between p\' and p (cm)')

        plt.show()
        plt.clf()
        plt.plot(range(len(losses_test)),losses_test,label="loss_val")
        plt.plot(range(len(losses_train)),losses_train,label="loss_train")
        plt.xlabel('epoch')
        plt.ylabel('mse loss')
        plt.show()
    # PATH = "./model_trained/net_param.pkl"
    # torch.save(fk_nn.state_dict(), PATH)
    # with open("./model_trained/min_max.txt","w") as wf:
    #     for min_q in output_range[0]:
    #         wf.write(str(min_q)+",")
    #     for i in range(6):
    #         wf.write(str(output_range[1][i]))
    #         if i != 5:
    #             wf.write(",")
    #         else:
    #             wf.write("\n")
    #     for min_p in input_range[0]:
    #         wf.write(str(min_p)+",")
    #     for i in range(3):
    #         wf.write(str(input_range[1][i]))
    #         if i != 2:
    #             wf.write(",")
    #         else:
    #             wf.write("\n")
    # torch.save(fk_nn, "./model_trained/net.pkl")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=500)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, \
                                help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, \
                                help='update steps for finetunning', default=10)
    argparser.add_argument('--hidden_size', type=int, help='update steps for finetunning', default=128)
    argparser.add_argument('--hidden_layer', type=int, help='update steps for finetunning', default=3)
    argparser.add_argument('--batch_size', type=int, help='update steps for finetunning', default=40)
    argparser.add_argument('--is_fk', type=bool, help='if is foward', default=False)
    argparser.add_argument('--g', type=bool, \
                                help='generate radom datas or using true data', default=False)
    argparser.add_argument('--file', type=str, \
                            help='file name', default="real_data")
    argparser.add_argument('--is_delta_model', type=bool, \
                                help='if needs to train a delta_p -> delta_q models', default=False)
    argparser.add_argument('--d', type=bool, \
                                help='if need draw pic', default=True)
    args = argparser.parse_args()
    main(args)