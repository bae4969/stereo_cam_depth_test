import cv2 as cv
import numpy as np
import tifffile
import MyCommon
import RawVideoFunc

My = MyCommon.MyCommon("D:/res", "STEREO_VIDEO/calibration_data", "TEST")

video_src_path = My.GetSrcFilePath("checker.mp4")

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

video_cap = RawVideoFunc.GetVideoCapcture(video_src_path)
if video_cap is None:
    print("Fail to open")
    exit(-1)


target_object_pts_list = []
left_object_pts_list = []
right_object_pts_list = []

frame_idx = 0
while True:
    ref_frame, color_left, color_right = RawVideoFunc.GetFrameWithIndex(
        video_cap, frame_idx
    )
    frame_idx += 30
    if ref_frame is False:
        break

    gray_left = cv.cvtColor(color_left, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(color_right, cv.COLOR_BGR2GRAY)

    ret_left, left_object_pts = cv.findChessboardCorners(
        gray_left, checkerboard_size, None
    )
    ret_right, right_object_pts = cv.findChessboardCorners(
        gray_right, checkerboard_size, None
    )
    if not ret_left or not ret_right:
        print("체스보드 코너를 찾지 못했습니다.")
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

video_cap.release()

##############################################################################################

rms1, K1, D1, _, _ = cv.calibrateCamera(
    target_object_pts_list, left_object_pts_list, image_size, None, None
)
rms2, K2, D2, _, _ = cv.calibrateCamera(
    target_object_pts_list, right_object_pts_list, image_size, None, None
)

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

T = np.array([[-78.5, 0, 0]], dtype=np.float64).T

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

left_map1, left_map2 = cv.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv.CV_32F)
right_map1, right_map2 = cv.initUndistortRectifyMap(
    K2, D2, R2, P2, image_size, cv.CV_32F
)

tifffile.imwrite(My.GetDstFilePath("T.tif"), T.astype(np.float32))
tifffile.imwrite(My.GetDstFilePath("P1.tif"), P1.astype(np.float32))
tifffile.imwrite(My.GetDstFilePath("left_map1.tif"), left_map1.astype(np.float32))
tifffile.imwrite(My.GetDstFilePath("left_map2.tif"), left_map2.astype(np.float32))
tifffile.imwrite(My.GetDstFilePath("right_map1.tif"), right_map1.astype(np.float32))
tifffile.imwrite(My.GetDstFilePath("right_map2.tif"), right_map2.astype(np.float32))

print("R matrix:\n", R)
print("T vector:", T.ravel())
