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

##############################################################################################

min_visualize_mm = 0
max_visualize_mm = 6000

fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

key_pressed = None
def on_key(event):
	global key_pressed
	key_pressed = event.key

fig.canvas.mpl_connect('key_press_event', on_key)

axs[0].set_title(f"left")
axs[1].set_title(f"right")

##############################################################################################

context = zmq.Context()
sub_socket = context.socket(zmq.SUB)
sub_socket.setsockopt(zmq.RCVHWM, 1)
sub_socket.setsockopt(zmq.RCVTIMEO, 10)
sub_socket.setsockopt_string(zmq.SUBSCRIBE, "STATE_CAMERA_SENSOR")
sub_socket.connect("tcp://192.168.135.32:45000")

poller = zmq.Poller()
poller.register(sub_socket, zmq.POLLIN)

while key_pressed != 'q':
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

    axs[0].clear()
    axs[1].clear()
    axs[0].imshow(color_left[:, :, [2, 1, 0]])
    axs[1].imshow(color_right[:, :, [2, 1, 0]])
    plt.tight_layout()
    plt.pause(0.001)
