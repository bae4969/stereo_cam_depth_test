import tifffile
import MyCommon

My = MyCommon.MyCommon("E:/res/STEREO_VIDEO", "captured_images", "calibration_data")

# 보정 데이터 로드
T = tifffile.imread(My.GetDstFilePath("T.tif"))
P1 = tifffile.imread(My.GetDstFilePath("P1.tif"))

focal_length = float(P1[0, 0])
baseline_mm = float(abs(T[0]))

print(f"카메라 정보:")
print(f"  Focal Length: {focal_length:.2f} pixels")
print(f"  Baseline: {baseline_mm:.2f} mm")

print(f"\n시차 계산 테스트:")
print(f"공식: disparity = (focal_length * baseline) / distance")

# 다양한 거리에서 시차 계산
distances = [100, 200, 500, 1000, 2000, 5000]  # mm 단위

for distance_mm in distances:
    disparity_pixels = (focal_length * baseline_mm) / distance_mm
    print(f"  거리: {distance_mm}mm -> 시차: {disparity_pixels:.2f} pixels")

print(f"\n시차에서 거리 계산 (역공식):")
print(f"공식: distance = (focal_length * baseline) / disparity")

# 예시 시차값들
disparities = [50, 25, 10, 5, 2.5, 1]  # pixels

for disparity_pixels in disparities:
    distance_mm = (focal_length * baseline_mm) / disparity_pixels
    print(f"  시차: {disparity_pixels} pixels -> 거리: {distance_mm:.0f}mm") 