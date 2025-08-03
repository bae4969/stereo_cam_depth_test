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
calibration_dir = My.GetDstFilePath("")

# 보정 데이터 로드
T = tifffile.imread(os.path.join(calibration_dir, "T.tif"))
P1 = tifffile.imread(os.path.join(calibration_dir, "P1.tif"))
left_map1 = tifffile.imread(os.path.join(calibration_dir, "left_map1.tif"))
left_map2 = tifffile.imread(os.path.join(calibration_dir, "left_map2.tif"))
right_map1 = tifffile.imread(os.path.join(calibration_dir, "right_map1.tif"))
right_map2 = tifffile.imread(os.path.join(calibration_dir, "right_map2.tif"))

image_size = (960, 720)
focal_length = float(P1[0, 0])
baseline_mm = float(abs(T[0]))

print(f"카메라 정보:")
print(f"  Focal Length: {focal_length:.2f} pixels")
print(f"  Baseline: {baseline_mm:.2f} mm")

##############################################################################################

# 저장된 이미지 파일들을 찾기
left_image_files = sorted(glob.glob(os.path.join(captured_images_dir, "left_*.png")))
right_image_files = sorted(glob.glob(os.path.join(captured_images_dir, "right_*.png")))

print(f"\n찾은 이미지 파일 수: {len(left_image_files)}")
if len(left_image_files) == 0:
    print("저장된 이미지가 없습니다. 먼저 monitoring.py로 이미지를 저장해주세요.")
    exit(-1)

print("\n" + "="*60)
print("스테레오 이미지 정합도 확인 프로그램")
print("="*60)
print("단축키 설명:")
print("  ↑/↓ : 거리 조정 (10mm 단위)")
print("  a/s : 거리 조정 (50mm 단위)")
print("  z/x : 거리 조정 (100mm 단위)")
print("  ←/→ : 이전/다음 이미지")
print("  r    : 거리 초기화")
print("  q    : 프로그램 종료")
print("="*60)

##############################################################################################

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
key_pressed = None

def on_key(event):
    global key_pressed
    key_pressed = event.key

fig.canvas.mpl_connect('key_press_event', on_key)

current_image_idx = 0
current_distance_mm = 200  # 초기 거리 20cm (매우 가까운 거리로 시작)

def calculate_disparity(distance_mm):
    """거리에 따른 시차 계산"""
    if distance_mm <= 0:
        return 0
    # 스테레오 시차 공식: disparity = (focal_length * baseline) / distance
    disparity_pixels = (focal_length * baseline_mm) / distance_mm
    return disparity_pixels

def shift_image_for_distance(image, disparity_pixels):
    """거리에 따라 이미지를 이동"""
    height, width = image.shape[:2]
    # 우측 이미지를 좌측으로 이동 (시차만큼)
    shift_matrix = np.float32([[1, 0, disparity_pixels], [0, 1, 0]])
    shifted_image = cv.warpAffine(image, shift_matrix, (width, height))
    return shifted_image

def update_display():
    """화면 업데이트"""
    if current_image_idx >= len(left_image_files):
        return
    
    # 이미지 로드
    left_image_path = left_image_files[current_image_idx]
    right_image_path = right_image_files[current_image_idx]
    
    color_left = cv.imread(left_image_path)
    color_right = cv.imread(right_image_path)
    
    if color_left is None or color_right is None:
        print(f"이미지를 읽을 수 없습니다: {left_image_path}, {right_image_path}")
        return
    
    # 보정된 이미지 생성
    rect_left = cv.remap(color_left, left_map1, left_map2, cv.INTER_LINEAR)
    rect_right = cv.remap(color_right, right_map1, right_map2, cv.INTER_LINEAR)
    
    # 현재 거리에 따른 시차 계산
    disparity_pixels = calculate_disparity(current_distance_mm)
    
    # 디버깅 정보 출력
    if current_image_idx == 0:  # 첫 번째 이미지에서만 출력
        print(f"디버깅 정보:")
        print(f"  Focal Length: {focal_length:.2f} pixels")
        print(f"  Baseline: {baseline_mm:.2f} mm")
        print(f"  Distance: {current_distance_mm} mm")
        print(f"  Calculated Disparity: {disparity_pixels:.2f} pixels")
    
    # 거리에 따라 우측 이미지 이동
    shifted_right = shift_image_for_distance(rect_right, disparity_pixels)
    
    # 이미지 겹치기 (좌측은 빨간색, 우측은 청록색)
    overlay = np.zeros_like(rect_left)
    overlay[:, :, 2] = rect_left[:, :, 2]  # 빨간색 채널 (좌측)
    overlay[:, :, 0] = shifted_right[:, :, 0]  # 파란색 채널 (우측)
    overlay[:, :, 1] = shifted_right[:, :, 1]  # 초록색 채널 (우측)
    
    # 스테레오 정규화 확인을 위한 선 그리기
    height, width = overlay.shape[:2]
    for i in range(0, height, 30):
        cv.line(overlay, (0, i), (width, i), (0, 255, 0), 1)
    
    # 결과 표시
    ax.clear()
    
    # 겹친 이미지 표시
    ax.imshow(overlay[:, :, [2, 1, 0]])
    ax.set_title(f"겹친 이미지 (빨강: 좌측, 청록: 우측) | "
                 f"거리: {current_distance_mm}mm | "
                 f"시차: {disparity_pixels:.1f}px", fontsize=14)
    
    # 정합도 정보를 제목에 추가
    diff = cv.absdiff(rect_left, shifted_right)
    match_score = 255 - np.mean(diff)
    match_percentage = (match_score / 255) * 100
    
    # 정보 표시
    fig.suptitle(f"이미지 {current_image_idx+1}/{len(left_image_files)} | "
                 f"정합도: {match_percentage:.1f}%", fontsize=12)
    
    plt.tight_layout()
    plt.pause(0.01)

# 초기 표시
update_display()

##############################################################################################

while key_pressed != 'q':
    key_pressed = None
    while key_pressed is None:
        plt.pause(0.01)
    
    if key_pressed == 'up':
        current_distance_mm += 10
        print(f"거리 증가: {current_distance_mm}mm (+10mm)")
        update_display()
    elif key_pressed == 'down':
        current_distance_mm = max(10, current_distance_mm - 10)
        print(f"거리 감소: {current_distance_mm}mm (-10mm)")
        update_display()
    elif key_pressed == 'ctrl+up':
        current_distance_mm += 50
        print(f"거리 증가: {current_distance_mm}mm (+50mm)")
        update_display()
    elif key_pressed == 'ctrl+down':
        current_distance_mm = max(10, current_distance_mm - 50)
        print(f"거리 감소: {current_distance_mm}mm (-50mm)")
        update_display()
    elif key_pressed == 'shift+up':
        current_distance_mm += 100
        print(f"거리 증가: {current_distance_mm}mm (+100mm)")
        update_display()
    elif key_pressed == 'shift+down':
        current_distance_mm = max(10, current_distance_mm - 100)
        print(f"거리 감소: {current_distance_mm}mm (-100mm)")
        update_display()
    elif key_pressed == 'left':
        current_image_idx = max(0, current_image_idx - 1)
        print(f"이전 이미지: {current_image_idx+1}/{len(left_image_files)}")
        update_display()
    elif key_pressed == 'right':
        current_image_idx = min(len(left_image_files) - 1, current_image_idx + 1)
        print(f"다음 이미지: {current_image_idx+1}/{len(left_image_files)}")
        update_display()
    elif key_pressed == 'r':
        current_distance_mm = 200
        print(f"거리 초기화: {current_distance_mm}mm")
        update_display()
    elif key_pressed == 'q':
        print("프로그램을 종료합니다.")
        break
    else:
        print(f"알 수 없는 키: {key_pressed}")

plt.close()
print("\n프로그램이 종료되었습니다.") 