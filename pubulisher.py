# -*- coding: utf-8 -*-  
"""
    description: JointPub Node
"""
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math
from dynamixel_msgs.msg import MotorStateList
from std_msgs.msg import Float64
import numpy as np
# <std_msgs/Float64.h>
class JointPub():
    def __init__(self):
        self.node_names = [
            #0
            "Nothing",
            #1
            "/RArmShoulderPitch_controller/command",
            #2
            "/LArmShoulderPitch_controller/command",
            #3
            "/RArmShoulderRoll_controller/command",
            #4
            "/LArmShoulderRoll_controller/command",
            #5
            "/RArmElbowYaw_controller/command",
            #6
            "/LArmElbowYaw_controller/command",
            #7
            "/RHipYaw_controller/command",
            #8
            "/LHipYaw_controller/command",
            #9
            "/RHipRoll_controller/command",
            #10
            "/LHipRoll_controller/command",
            #11
            "/RHipPitch_controller/command",
            #12
            "/LHipPitch_controller/command",
            #13
            "/RKneePitch_controller/command",
            #14
            "/LKneePitch_controller/command",
            #15
            "/RAnklePitch_controller/command",
            #16
            "/LAnklePitch_controller/command",
            #17
            "/RFootRoll_controller/command",
            #18
            "/LFootRoll_controller/command",
            #19
            "/NeckYaw_controller/command",
            #20
            "/HeadPitch_controller/command",
            #21
            "/RArmElbowRoll_controller/command",
            #22
            "/LArmElbowRoll_controller/command",
            #23
            "/RArmWristYaw_controller/command",
            #24
            "/LArmWristYaw_controller/command",
            #25
            "/RArmWristRoll_controller/command",
            #26
            "/LArmWristRoll_controller/command",
            #27
            "/RArmHand_controller/command",
            #28
            "/LArmHand_controller/command"
            ]
        self.pub = []
        rospy.init_node('jointStatePublisher')
        self.rate = rospy.Rate(100) # 100hz
        for i in range(0,28):
            self.pub.append(rospy.Publisher(self.node_names[i], Float64, queue_size=10))

    
    def publish_sigle_joint(self,joint_value,joint_index):
        D2R = np.pi/180        
        pub_str = Float64()
        pub_str.data = joint_value
        if joint_value > 3.14 or joint_value < -3.14:
            joint_value = joint_value*D2R
        pub_str.data = joint_value
        # print("pub angle ",joint_value," to ",joint_index)
        
        for i in range(20):
            self.pub[joint_index].publish(pub_str)
            self.rate.sleep()

    def publish_sigle_jointD(self,joint_value,joint_index):
        D2R = np.pi/180
        pub_str = Float64()
        pub_str.data = joint_value*D2R
        # print("pub angle ",joint_value*D2R," to ",joint_index)        
        for i in range(20):
            self.pub[joint_index].publish(pub_str)
            self.rate.sleep()
    
    def publish_jointsR(self,joint_values):
        joint_indexs = [1,3,5,21,23,25]
        for joint_value,joint_index in zip(joint_values,joint_indexs):
            self.publish_sigle_joint(joint_value,joint_index)
    
    def publish_jointsD(self,joint_values):
        D2R = np.pi/180
        joint_values = [joint * D2R for joint in joint_values]
        self.publish_jointsR(joint_values)
    