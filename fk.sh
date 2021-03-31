python init_robot_position.py
sudo modprobe uvcvideo
rosrun realsense_camera realsense-capture
python3 get_fkmodel.py --file frame_data_0301