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
        with open("/home/pku-hr6/yyf_ws/data/arm_running.txt",'r') as rf:
            lines = rf.readlines()
            for line in lines:
                datas = line.split(" ")
                datas = [float(data) for data in datas]
                print(datas)
                tracking_datas.append(datas)
    else:
        rospy.init_node('data_saver')
        

    data_nums = args.n
    file_names = args.file
    datas = []
    save_files = os.path.join("./data",file_names+'.txt')
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    pc = rs.pointcloud()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    device_product_line = str(device.get_info(rs.camera_info.product_line))
    # device_product_line is SR300

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # get scale of depth sensor
    depth_sensor = pipeline_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()


    # clipping_distance_in_meters meters away
    clipping_distance_in_meters = 0.5 # 0.5 meter
    clipping_distance = clipping_distance_in_meters/depth_scale

    # creat align map
    align_to = rs.stream.color
    align = rs.align(align_to)

    l_h = 170
    l_s = 42
    l_v = 41
    u_h = 185
    u_s = 255
    u_v = 255
    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])
    count = 0
    with open(save_files,"w") as wf:
        # while count < data_nums:
        while len(datas) < data_nums:
        # while True:
            # Wait for a coherent pair of frames: depth and color
            if args.t:
                if count>=len(tracking_datas) - 1:
                    break
            
            frames = pipeline.wait_for_frames()
            aligned_frame = align.process(frames)

            depth_frame = aligned_frame.get_depth_frame()
            color_frame = aligned_frame.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # remove background
            grey_color = 153
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image))   # depth img is 1 channel, color is 3 channels
            bg_rmvd = np.where((depth_image_3d > clipping_distance)|(depth_image_3d<=0),grey_color,color_image)
            # get final img
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,alpha=0.03),cv2.COLORMAP_JET)
            
            hsv_map = cv2.cvtColor(bg_rmvd,cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv_map, l_b, u_b)
            res = cv2.bitwise_and(bg_rmvd, bg_rmvd, mask=mask)

            img = np.hstack((bg_rmvd, depth_colormap))

            # get object from mask map and calculate position
            mask_index = np.nonzero(mask)

            valid_p = False
            depth_point = np.array([])
            if not mask_index[0].shape[0] == 0:
                valid_p = True
                x_index = int(np.median(mask_index[1]))
                y_index = int(np.median(mask_index[0]))            
                x_min = x_index - 20
                x_max = x_index + 20
                y_min = y_index - 20
                y_max = y_index + 20
                # Intrinsics & Extrinsics
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                # print(depth_intrin)
                #  640x480  p[314.696 243.657]  f[615.932 615.932]
                depth_pixel = [x_index,y_index]
                dist2obj = depth_frame.get_distance(x_index,y_index)
                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dist2obj)
                depth_point = [float(dep) * 100 for dep in depth_point]    # using cm
                depth_point = [depth_point[2], -depth_point[0],-depth_point[1]] #[x,y,z](in base) = [z,-x,-y](in camera)
                depth_point = [depth_point[0]+1.8, depth_point[1], depth_point[2]+5.3]
                txt = "({:.2f},{:.2f},{:.2f})".format(depth_point[0],depth_point[1],depth_point[2])
                # print(txt)
                cv2.rectangle(res, (x_min,y_min),(x_max,y_max),(255,0,0),2)
                cv2.putText(res, txt, (x_index,y_index), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
            cv2.imshow('Align Example',img)
            cv2.imshow("mask", mask)
            cv2.imshow("res", res)
            # print(depth_point)

            msg = rospy.wait_for_message('/motor_states/pan_tilt_port',MotorStateList,timeout=10)
            # print(len(msg.motor_states))
            joints,valid_q = get_joint_angle(msg)
            if valid_p and valid_q:
                depth_point = np.array(depth_point)
                # print(joints)
                # print(depth_point)
                if args.only_q:
                    data = joints
                else:   
                    data = np.append(depth_point, joints)
                print(data)
                datas.append(data)
            if args.t: # pubulish joint
                data = tracking_datas[count]
                Pubs.publish_jointsD(data)
                count = count + 1
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
    argparser.add_argument('--file', type=str, help='file name(or path)', default="arm_running_300")
    argparser.add_argument('--only_q', type=bool, help='type of save datas, only q or p&q', default=False)
    argparser.add_argument('--t', type=bool, help='read track from file', default=False)
    
    args = argparser.parse_args()
    main(args)