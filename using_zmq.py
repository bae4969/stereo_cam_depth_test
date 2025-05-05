import cv2 as cv
import numpy as np
import zmq
import math
import tifffile
import numpy as np
from matplotlib import pyplot as plt
from MyCommon import MyCommon
import RawVideoFunc

My = MyCommon("D:/res", "STEREO_VIDEO", "TEST")

video_src_path = My.GetSrcFilePath("moving.mp4")

T = tifffile.imread(My.GetSrcFilePath("calibration_data/T.tif"))
P1 = tifffile.imread(My.GetSrcFilePath("calibration_data/P1.tif"))
left_map1 = tifffile.imread(My.GetSrcFilePath("calibration_data/left_map1.tif"))
left_map2 = tifffile.imread(My.GetSrcFilePath("calibration_data/left_map2.tif"))
right_map1 = tifffile.imread(My.GetSrcFilePath("calibration_data/right_map1.tif"))
right_map2 = tifffile.imread(My.GetSrcFilePath("calibration_data/right_map2.tif"))

image_size = (960, 720)
focal_length = P1[0, 0]
baseline_mm = abs(T[0])
min_depth_mm = 1000
max_disp = (focal_length * baseline_mm) / min_depth_mm
num_disp = int(math.ceil(max_disp / 16.0)) * 16

min_disp = 0
num_channels = 1
block_size = 11

stereo = cv.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * num_channels * block_size**2,
    P2=32 * num_channels * block_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=1,
)

##############################################################################################

context = zmq.Context()
sub_socket = context.socket(zmq.SUB)
sub_socket.setsockopt(zmq.RCVHWM, 1)
sub_socket.setsockopt(zmq.RCVTIMEO, 10)
sub_socket.setsockopt_string(zmq.SUBSCRIBE, "STATE_CAMERA_SENSOR")  # 토픽 필터
sub_socket.connect("tcp://135.135.135.32:45000")

poller = zmq.Poller()
poller.register(sub_socket, zmq.POLLIN)

while True:
    msg_parts = None
    while True:
        try:
            socks = dict(poller.poll(timeout=5))
            if sub_socket in socks:
                msg_parts = sub_socket.recv_multipart(flags=zmq.NOBLOCK)
            else:
                break
        except zmq.Again:
            break

    if not msg_parts or len(msg_parts) < 3:
        continue

    color_left = cv.imdecode(np.frombuffer(msg_parts[1], dtype=np.uint8), cv.IMREAD_COLOR)
    color_right = cv.imdecode(np.frombuffer(msg_parts[2], dtype=np.uint8), cv.IMREAD_COLOR)
    if color_left is None or color_right is None:
        continue

    left_rect = cv.remap(color_left, left_map1, left_map2, cv.INTER_LINEAR)
    right_rect = cv.remap(color_right, right_map1, right_map2, cv.INTER_LINEAR)
    
    if num_channels is 1:
        left_rect = cv.cvtColor(left_rect, cv.COLOR_RGB2GRAY)
        right_rect = cv.cvtColor(right_rect, cv.COLOR_RGB2GRAY)

    disparity = stereo.compute(left_rect, right_rect).astype(np.float32) / 16.0
    disparity = disparity[:,num_disp:]
        
    depth_map = (focal_length * baseline_mm) / disparity
    depth_map[(depth_map < 0) | (5000 < depth_map)] = np.nan

    plt.imshow(depth_map, cmap='jet', vmin=0, vmax=5000)
    plt.title('depth')
    plt.colorbar()
    plt.pause(0.001)
    plt.clf()
    cv.imshow('left', left_rect)
    cv.imshow('right', right_rect)
    key = cv.waitKey(1)
    if key == 'q':
        break