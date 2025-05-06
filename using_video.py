import cv2 as cv
import numpy as np
import math
import tifffile
import numpy as np
from matplotlib import pyplot as plt
from MyCommon import MyCommon
import RawVideoFunc

My = MyCommon("D:/res", "STEREO_VIDEO", "TEST")

video_src_path = My.GetSrcFilePath("250505/record_20250505_1419.mp4")
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
block_size = 9

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

min_visualize_mm = 0
max_visualize_mm = 6000

fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=200)

key_pressed = None
def on_key(event):
	global key_pressed
	key_pressed = event.key

fig.canvas.mpl_connect('key_press_event', on_key)

axs[0].set_title(f"left")
axs[1].set_title(f"right")
axs[2].set_title(f"depth")

im = axs[2].imshow(np.zeros((100, 100)), cmap='inferno', vmin=min_visualize_mm, vmax=max_visualize_mm)
cbar = fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04, label="Dist (mm)")

##############################################################################################

video = RawVideoFunc.GetVideoCapcture(video_src_path)
if video is None:
    print('Fail to open video file!')
    exit(-1)

running_state = False
cur_idx = -1
nxt_idx = 0
max_idx = RawVideoFunc.GetMaxFrameIndex(video)
while key_pressed != 'q':
    if key_pressed == ' ':
        running_state = not running_state

    if key_pressed == 'left':
        nxt_idx = max(cur_idx - 10, 0)
    elif key_pressed == 'alt+left':
        nxt_idx = max(cur_idx - 1, 0)
    elif key_pressed == 'ctrl+left':
        nxt_idx = max(cur_idx - 30, 0)
    elif key_pressed == 'shift+left':
        nxt_idx = max(cur_idx - 300, 0)
    elif key_pressed == 'ctrl+shift+left':
        nxt_idx = max(cur_idx - 3000, 0)
    elif key_pressed == 'right':
        nxt_idx = min(cur_idx + 10, max_idx)
    elif key_pressed == 'alt+right':
        nxt_idx = min(cur_idx + 1, max_idx)
    elif key_pressed == 'ctrl+right':
        nxt_idx = min(cur_idx + 30, max_idx)
    elif key_pressed == 'shift+right':
        nxt_idx = min(cur_idx + 300, max_idx)
    elif key_pressed == 'ctrl+shift+right':
        nxt_idx = min(cur_idx + 3000, max_idx)
    elif running_state is True:
        nxt_idx = min(cur_idx + 10, max_idx)
    else:
        nxt_idx = max(cur_idx, 0)

    key_pressed = None
    if nxt_idx == cur_idx:
        plt.pause(1)
        continue

    ret, color_left, color_right = RawVideoFunc.GetFrameWithIndex(video, nxt_idx)
    cur_idx = nxt_idx
    if ret is False:
        plt.pause(1)
        continue

    left_rect = cv.remap(color_left, left_map1, left_map2, cv.INTER_LINEAR)
    right_rect = cv.remap(color_right, right_map1, right_map2, cv.INTER_LINEAR)
    
    if num_channels == 1:
        left_input = cv.cvtColor(left_rect, cv.COLOR_RGB2GRAY)
        right_input = cv.cvtColor(right_rect, cv.COLOR_RGB2GRAY)
    else:
        left_input = left_rect
        right_input = right_rect

    disparity = stereo.compute(left_input, right_input).astype(np.float32) / 16.0
    disparity = disparity[:,num_disp:]
        
    depth_map = (focal_length * baseline_mm) / disparity
    depth_map[(depth_map < min_visualize_mm) | (max_visualize_mm < depth_map)] = np.nan

    axs[0].clear()
    axs[1].clear()
    axs[2].clear()
    axs[0].imshow(left_rect[:, :, [2, 1, 0]])
    axs[1].imshow(right_rect[:, :, [2, 1, 0]])
    im = axs[2].imshow(depth_map, cmap='inferno', vmin=min_visualize_mm, vmax=max_visualize_mm)
    cbar.update_normal(im)
    plt.tight_layout()
    plt.pause(0.001)
