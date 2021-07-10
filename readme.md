my workspace :
	/home/pku-hr6/yyf_ws/src/meta_control_model
pku-hr6.0 workspcace :
	/home/pku-hr6/catkin_ws/src


# robot control:
in /home/pku-hr6/catkin_ws/src/my_dynamixel_tutorial

# file ability:

get_fkmodel.py:
	get D-H param of robot
train_***.py:
	using network to model robot


cd yyf_ws/src/meta_control_model

roslaunch my_dynamixel_tutorial controller_manager.launch

# in another terminal, start tilt controller
roslaunch my_dynamixel_tutorial start_tilt_controller.launch
(if get "done",succed!)


# init robot's position, take care !
python init_robot_position.py 		

## start the detection program
sudo modprobe uvcvideo 				#first start, you should enter this commmand and the password is "robot"
rosrun realsense_camera get_hand_position

#if change link length

#using contorl model to postion_tgt

python test_delta_ik_model.py 
