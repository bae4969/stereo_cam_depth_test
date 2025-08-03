import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import tifffile
import MyCommon
import os
import glob

My = MyCommon.MyCommon("E:/res/STEREO_VIDEO", "captured_images", "calibration_data")

# 저장된 이미지들이 있는 디렉토리
captured_images_dir = My.GetSrcFilePath("")

image_size = (960, 720)
checkerboard_size = (10, 7)
square_size = 25
target_object_pts = np.zeros(
    (checkerboard_size[0] * checkerboard_size[1], 3), np.float32
)
target_object_pts[:, :2] = np.mgrid[
    0 : checkerboard_size[0], 0 : checkerboard_size[1]
].T.reshape(-1, 2)
target_object_pts *= square_size

##############################################################################################

fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

key_pressed = None
def on_key(event):
	global key_pressed
	key_pressed = event.key

fig.canvas.mpl_connect('key_press_event', on_key)

axs[0].set_title(f"left")
axs[1].set_title(f"right")

##############################################################################################

# 저장된 이미지 파일들을 찾기
left_image_files = sorted(glob.glob(os.path.join(captured_images_dir, "left_*.png")))
right_image_files = sorted(glob.glob(os.path.join(captured_images_dir, "right_*.png")))

print(f"찾은 이미지 파일 수: {len(left_image_files)}")
if len(left_image_files) == 0:
    print("저장된 이미지가 없습니다. 먼저 monitoring.py로 이미지를 저장해주세요.")
    exit(-1)

print("\n" + "="*60)
print("스테레오 카메라 보정 프로그램")
print("="*60)
print("단축키 설명:")
print("  ↑ 또는 → : 현재 이미지를 보정에 추가")
print("  ↓ 또는 ← : 현재 이미지를 스킵")
print("  a        : 모든 이미지를 보정에 추가")
print("  q        : 보정 작업 중단")
print("="*60)
print("체스보드 코너가 감지된 이미지만 표시됩니다.")
print("보정에는 최소 5장의 이미지가 필요합니다.")
print("="*60 + "\n")

target_object_pts_list = []
left_object_pts_list = []
right_object_pts_list = []

image_idx = 0
add_all_mode = False

while image_idx < len(left_image_files):
    # 이미지 파일 읽기
    left_image_path = left_image_files[image_idx]
    right_image_path = right_image_files[image_idx]
    
    color_left = cv.imread(left_image_path)
    color_right = cv.imread(right_image_path)
    
    if color_left is None or color_right is None:
        print(f"이미지를 읽을 수 없습니다: {left_image_path}, {right_image_path}")
        image_idx += 1
        continue

    gray_left = cv.cvtColor(color_left, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(color_right, cv.COLOR_BGR2GRAY)

    ret_left, left_object_pts = cv.findChessboardCorners(
        gray_left, checkerboard_size, None
    )
    ret_right, right_object_pts = cv.findChessboardCorners(
        gray_right, checkerboard_size, None
    )
    
    if not ret_left or not ret_right:
        print(f"[{image_idx}] 체스보드 코너를 찾지 못했습니다.")
        image_idx += 1
        continue

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    left_object_pts = cv.cornerSubPix(
        gray_left, left_object_pts, (11, 11), (-1, -1), criteria
    )
    right_object_pts = cv.cornerSubPix(
        gray_right, right_object_pts, (11, 11), (-1, -1), criteria
    )
    
    # 코너를 그린 이미지 생성
    color_left_with_corners = color_left.copy()
    color_right_with_corners = color_right.copy()
    cv.drawChessboardCorners(color_left_with_corners, checkerboard_size, left_object_pts, ret_left)
    cv.drawChessboardCorners(color_right_with_corners, checkerboard_size, right_object_pts, ret_right)
    
    axs[0].clear()
    axs[1].clear()
    axs[0].imshow(color_left_with_corners[:, :, [2, 1, 0]])
    axs[1].imshow(color_right_with_corners[:, :, [2, 1, 0]])
    axs[0].set_title(f"left - {os.path.basename(left_image_path)}")
    axs[1].set_title(f"right - {os.path.basename(right_image_path)}")
    plt.tight_layout()
    plt.pause(0.01)

    # 키 입력 대기
    key_pressed = None
    while key_pressed is None:
        plt.pause(0.01)

    is_add_frame = False
    
    if key_pressed == 'up' or key_pressed == 'right':
        is_add_frame = True
        print(f"[{image_idx}] 추가됨")
    elif key_pressed == 'down' or key_pressed == 'left':
        is_add_frame = False
        print(f"[{image_idx}] 스킵됨")
    elif key_pressed == 'a':
        add_all_mode = True
        is_add_frame = True
        print(f"[{image_idx}] 추가됨 (모든 이미지 추가 모드 활성화)")
    elif key_pressed == 'q':
        print("보정 작업을 중단합니다.")
        break
    else:
        print(f"알 수 없는 키: {key_pressed}")
        continue

    if is_add_frame:
        target_object_pts_list.append(target_object_pts)
        left_object_pts_list.append(left_object_pts)
        right_object_pts_list.append(right_object_pts)

    image_idx += 1
    
    # 모든 이미지 추가 모드인 경우 자동으로 다음 이미지들도 추가
    if add_all_mode:
        while image_idx < len(left_image_files):
            left_image_path = left_image_files[image_idx]
            right_image_path = right_image_files[image_idx]
            
            color_left = cv.imread(left_image_path)
            color_right = cv.imread(right_image_path)
            
            if color_left is None or color_right is None:
                print(f"이미지를 읽을 수 없습니다: {left_image_path}, {right_image_path}")
                image_idx += 1
                continue

            gray_left = cv.cvtColor(color_left, cv.COLOR_BGR2GRAY)
            gray_right = cv.cvtColor(color_right, cv.COLOR_BGR2GRAY)

            ret_left, left_object_pts = cv.findChessboardCorners(
                gray_left, checkerboard_size, None
            )
            ret_right, right_object_pts = cv.findChessboardCorners(
                gray_right, checkerboard_size, None
            )
            
            if not ret_left or not ret_right:
                print(f"[{image_idx}] 체스보드 코너를 찾지 못했습니다.")
                image_idx += 1
                continue

            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            left_object_pts = cv.cornerSubPix(
                gray_left, left_object_pts, (11, 11), (-1, -1), criteria
            )
            right_object_pts = cv.cornerSubPix(
                gray_right, right_object_pts, (11, 11), (-1, -1), criteria
            )
            
            target_object_pts_list.append(target_object_pts)
            left_object_pts_list.append(left_object_pts)
            right_object_pts_list.append(right_object_pts)
            
            print(f"[{image_idx}] 자동 추가됨")
            image_idx += 1
        
        print(f"총 {len(target_object_pts_list)}장의 이미지가 추가되었습니다.")
        break

plt.close()

##############################################################################################

if len(target_object_pts_list) < 5:
    print("보정에 필요한 최소 이미지 수(5장)가 부족합니다.")
    exit(-1)

print(f"총 {len(target_object_pts_list)}장의 이미지로 보정을 시작합니다.")

# 개별 카메라 보정
rms1, K1, D1, _, _ = cv.calibrateCamera(
    target_object_pts_list, left_object_pts_list, image_size, None, None
)
rms2, K2, D2, _, _ = cv.calibrateCamera(
    target_object_pts_list, right_object_pts_list, image_size, None, None
)

print(f"Left camera RMS: {rms1}")
print(f"Right camera RMS: {rms2}")

# 스테레오 보정
_, _, _, _, _, R, T, E, F = cv.stereoCalibrate(
    target_object_pts_list,
    left_object_pts_list,
    right_object_pts_list,
    K1,
    D1,
    K2,
    D2,
    image_size,
    criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
    flags=(
        cv.CALIB_USE_INTRINSIC_GUESS
        | cv.CALIB_FIX_ASPECT_RATIO
        | cv.CALIB_SAME_FOCAL_LENGTH
        | cv.CALIB_ZERO_TANGENT_DIST
    ),
)

# 스테레오 정규화
R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
    K1,
    D1,
    K2,
    D2,
    image_size,
    R,
    T,
    flags=cv.CALIB_ZERO_DISPARITY,
    alpha=0,  # 여백 제거
)

# 리매핑 맵 생성
left_map1, left_map2 = cv.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv.CV_32F)
right_map1, right_map2 = cv.initUndistortRectifyMap(
    K2, D2, R2, P2, image_size, cv.CV_32F
)

# 결과 저장
calibration_dir = My.GetDstFilePath("")
os.makedirs(calibration_dir, exist_ok=True)

tifffile.imwrite(os.path.join(calibration_dir, "T.tif"), T.astype(np.float32))
tifffile.imwrite(os.path.join(calibration_dir, "P1.tif"), P1.astype(np.float32))
tifffile.imwrite(os.path.join(calibration_dir, "left_map1.tif"), left_map1.astype(np.float32))
tifffile.imwrite(os.path.join(calibration_dir, "left_map2.tif"), left_map2.astype(np.float32))
tifffile.imwrite(os.path.join(calibration_dir, "right_map1.tif"), right_map1.astype(np.float32))
tifffile.imwrite(os.path.join(calibration_dir, "right_map2.tif"), right_map2.astype(np.float32))

print("보정 완료!")
print("R matrix:\n", R)
print("T vector:", T.ravel())
print(f"보정 결과가 {calibration_dir}에 저장되었습니다.")

##############################################################################################
# 보정 결과 확인
print("\n" + "="*60)
print("보정 결과 확인")
print("="*60)
print("보정된 이미지들을 확인하려면 'y'를 누르세요.")
print("확인을 건너뛰려면 'q'를 누르세요.")

# 새로운 figure와 키 이벤트 핸들러 생성
fig_result = plt.figure(figsize=(15, 5))
key_pressed = None

def on_key_result(event):
    global key_pressed
    key_pressed = event.key

fig_result.canvas.mpl_connect('key_press_event', on_key_result)

# 키 입력 대기
while key_pressed is None:
    plt.pause(0.01)

if key_pressed == 'q':
    print("보정 결과 확인을 건너뜁니다.")
    plt.close(fig_result)
elif key_pressed == 'y':
    print("보정된 이미지들을 표시합니다...")
    
    # 보정에 사용된 이미지들의 인덱스 찾기
    used_image_indices = []
    for i, (left_pts, right_pts) in enumerate(zip(left_object_pts_list, right_object_pts_list)):
        if len(left_pts) > 0 and len(right_pts) > 0:
            used_image_indices.append(i)
    
    if len(used_image_indices) > 0:
        print(f"총 {len(used_image_indices)}장의 보정된 이미지를 확인할 수 있습니다.")
        print("단축키: ←/→ (이전/다음), q (종료)")
        
        current_idx = 0
        while True:
            # 현재 이미지 로드
            test_idx = used_image_indices[current_idx]
            left_image_path = left_image_files[test_idx]
            right_image_path = right_image_files[test_idx]
            
            color_left = cv.imread(left_image_path)
            color_right = cv.imread(right_image_path)
            
            if color_left is not None and color_right is not None:
                # 보정된 이미지 생성
                rect_left = cv.remap(color_left, left_map1, left_map2, cv.INTER_LINEAR)
                rect_right = cv.remap(color_right, right_map1, right_map2, cv.INTER_LINEAR)
                
                # 스테레오 정규화 확인을 위한 선 그리기
                height, width = rect_left.shape[:2]
                for i in range(0, height, 30):
                    cv.line(rect_left, (0, i), (width, i), (0, 255, 0), 1)
                    cv.line(rect_right, (0, i), (width, i), (0, 255, 0), 1)
                
                # 스테레오 매칭을 위한 그레이스케일 변환
                gray_left = cv.cvtColor(rect_left, cv.COLOR_BGR2GRAY)
                gray_right = cv.cvtColor(rect_right, cv.COLOR_BGR2GRAY)
                
                # 스테레오 매칭
                stereo = cv.StereoSGBM_create(
                    minDisparity=0,
                    numDisparities=128,
                    blockSize=5,
                    P1=8 * 3 * 5**2,
                    P2=32 * 3 * 5**2,
                    disp12MaxDiff=1,
                    uniquenessRatio=10,
                    speckleWindowSize=100,
                    speckleRange=32
                )
                
                disparity = stereo.compute(gray_left, gray_right)
                disparity = disparity.astype(np.float32) / 16.0
                
                # 결과 표시
                plt.clf()
                axs_result = fig_result.subplots(1, 3)
                
                # 보정된 이미지와 depth 맵 표시
                axs_result[0].imshow(rect_left[:, :, [2, 1, 0]])
                axs_result[0].set_title(f"보정 후 - 좌측 ({current_idx+1}/{len(used_image_indices)})")
                axs_result[1].imshow(rect_right[:, :, [2, 1, 0]])
                axs_result[1].set_title(f"보정 후 - 우측 ({current_idx+1}/{len(used_image_indices)})")
                
                # Depth 맵 표시
                depth_map = axs_result[2].imshow(disparity, cmap='plasma')
                axs_result[2].set_title(f"Depth Map ({current_idx+1}/{len(used_image_indices)})")
                plt.colorbar(depth_map, ax=axs_result[2], shrink=0.8)
                
                plt.tight_layout()
                plt.pause(0.01)
                
                # 키 입력 대기
                key_pressed = None
                while key_pressed is None:
                    plt.pause(0.01)
                
                if key_pressed == 'left':
                    current_idx = max(0, current_idx - 1)
                elif key_pressed == 'right':
                    current_idx = min(len(used_image_indices) - 1, current_idx + 1)
                elif key_pressed == 'q':
                    break
                else:
                    print(f"알 수 없는 키: {key_pressed}")
            
        print("보정 결과 확인 완료!")
        print("- 보정 후 이미지: 왜곡이 제거되고 정규화된 스테레오 이미지")
        print("- 녹색 선: 스테레오 정규화 확인용 (수평선이 일치해야 함)")
        print("- Depth 맵: 스테레오 매칭을 통한 깊이 정보 (plasma 컬러맵)")

print("\n프로그램을 종료합니다.") 
