import cv2 as cv
import numpy as np
import zmq
import math
import tifffile
import numpy as np
from matplotlib import pyplot as plt
from MyCommon import MyCommon
import RawVideoFunc
import os
from datetime import datetime

My = MyCommon("E:/res/STEREO_VIDEO", "", "captured_images")

##############################################################################################

min_visualize_mm = 0
max_visualize_mm = 6000

fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

key_pressed = None
save_counter = 0
save_dir = My.GetDstFilePath("")
os.makedirs(save_dir, exist_ok=True)

def on_key(event):
	global key_pressed, save_counter, color_left, color_right
	key_pressed = event.key
	
	if event.key == 'c':
		# 현재 이미지들을 저장
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		left_filename = os.path.join(save_dir, f"left_{timestamp}_{save_counter:04d}.png")
		right_filename = os.path.join(save_dir, f"right_{timestamp}_{save_counter:04d}.png")
		
		if color_left is not None and color_right is not None:
			cv.imwrite(left_filename, color_left)
			cv.imwrite(right_filename, color_right)
			save_counter += 1
			print(f"이미지 저장됨: {left_filename}, {right_filename}")

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

# 전역 변수로 현재 이미지들을 저장
color_left = None
color_right = None

print("프로그램 시작 - 'c' 키를 누르면 이미지가 저장됩니다. 'q' 키를 누르면 종료됩니다.")

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

print(f"총 {save_counter}장의 이미지가 저장되었습니다.")
