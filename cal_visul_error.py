# calculate visul data error

import numpy as np
from fk_models import cal_dis


def main():
    str_ = "(38.66,-17.25,23.21)|(38.36,-16.66,19.90)|(38.33,-1.82,11.38)|(38.25,-4.50,11.29)|(38.10,-7.19,11.29)|(37.82,-9.78,11.10)|(37.78,-12.62,11.17)|(36.95,-27.67,11.65)|(36.73,-31.41,11.82)|(37.37,-15.29,8.32)|(37.15,-14.97,5.67)|(36.98,-14.87,3.01)|(11.06,-4.35,-0.75)|(36.77,-14.54,-2.48)"
    # str_ = "(38.66,-17.25,23.21)|(38.36,-16.66,19.90)|(38.10,-7.19,11.29)|(37.82,-9.78,11.10)|(37.78,-12.62,11.17)|(37.37,-15.29,8.32)"
    datas = str_.split('|')
    points = []
    for p in datas:
        nums = p[1:-1].split(',')
        # print(nums)
        nums = [float(num) for num in nums]
        points.append(nums)
    points = np.array(points)
    dis = []
    for p in points:
        for p2 in points:
            dis_12 = cal_dis(p,p2)
            if dis_12 == 0:
                continue
            if dis_12<3.5:
                dis.append(dis_12)
    dis.sort()    
    dis = np.array(dis)
    print("mean dis:",dis.mean())
    print(dis)
if __name__ == '__main__':
    main()