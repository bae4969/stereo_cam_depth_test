import cv2 as cv
import numpy as np
import zmq
import math
import tifffile
from matplotlib import pyplot as plt
from MyCommon import MyCommon
import os
from datetime import datetime
import glob

class StereoRealtimeCalibration:
    def __init__(self):
        # MyCommon 초기화
        self.My = MyCommon("E:/res/STEREO_VIDEO", "", "captured_images")
        
        # 보정 파라미터
        self.image_size = (960, 720)
        self.checkerboard_size = (10, 7)
        self.square_size = 25
        self.target_object_pts = np.zeros(
            (self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32
        )
        self.target_object_pts[:, :2] = np.mgrid[
            0 : self.checkerboard_size[0], 0 : self.checkerboard_size[1]
        ].T.reshape(-1, 2)
        self.target_object_pts *= self.square_size
        
        # ZMQ 설정
        self.context = zmq.Context()
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.setsockopt(zmq.RCVHWM, 1)
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 10)
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "STATE_CAMERA_SENSOR")
        self.sub_socket.connect("tcp://192.168.135.32:45000")
        
        self.poller = zmq.Poller()
        self.poller.register(self.sub_socket, zmq.POLLIN)
        
        # 저장 관련 변수
        self.save_counter = 0
        self.save_dir = self.My.GetDstFilePath("")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 보정 관련 변수
        self.target_object_pts_list = []
        self.left_object_pts_list = []
        self.right_object_pts_list = []
        self.calibration_completed = False
        
        # GUI 설정
        self.setup_gui()
        
    def setup_gui(self):
        """GUI 초기화"""
        self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # 서브플롯 제목 설정
        self.axs[0].set_title("좌측 이미지")
        self.axs[1].set_title("우측 이미지")
        self.axs[2].set_title("체스보드 감지")
        
        self.key_pressed = None
        
        print("\n" + "="*60)
        print("실시간 스테레오 카메라 보정 프로그램")
        print("="*60)
        print("단축키 설명:")
        print("  c    : 현재 이미지 저장")
        print("  b    : 보정 시작 (저장된 이미지들로)")
        print("  v    : 보정 결과 확인")
        print("  q    : 프로그램 종료")
        print("="*60)
        print("1. 'c' 키로 체스보드 이미지들을 저장하세요.")
        print("2. 충분한 이미지가 저장되면 'b' 키로 보정을 시작하세요.")
        print("3. 보정 완료 후 'v' 키로 결과를 확인하세요.")
        print("="*60)
        
    def on_key(self, event):
        """키보드 이벤트 처리"""
        self.key_pressed = event.key
        
    def save_current_images(self, color_left, color_right):
        """현재 이미지들을 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        left_filename = os.path.join(self.save_dir, f"left_{timestamp}_{self.save_counter:04d}.png")
        right_filename = os.path.join(self.save_dir, f"right_{timestamp}_{self.save_counter:04d}.png")
        
        cv.imwrite(left_filename, color_left)
        cv.imwrite(right_filename, color_right)
        self.save_counter += 1
        print(f"이미지 저장됨: {left_filename}, {right_filename}")
        
    def detect_chessboard(self, color_left, color_right):
        """체스보드 코너 감지"""
        gray_left = cv.cvtColor(color_left, cv.COLOR_BGR2GRAY)
        gray_right = cv.cvtColor(color_right, cv.COLOR_BGR2GRAY)
        
        ret_left, left_object_pts = cv.findChessboardCorners(
            gray_left, self.checkerboard_size, None
        )
        ret_right, right_object_pts = cv.findChessboardCorners(
            gray_right, self.checkerboard_size, None
        )
        
        if ret_left and ret_right:
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
            cv.drawChessboardCorners(color_left_with_corners, self.checkerboard_size, left_object_pts, ret_left)
            cv.drawChessboardCorners(color_right_with_corners, self.checkerboard_size, right_object_pts, ret_right)
            
            return True, color_left_with_corners, color_right_with_corners, left_object_pts, right_object_pts
        else:
            return False, color_left, color_right, None, None
            
    def update_display(self, color_left, color_right, chessboard_detected=False):
        """화면 업데이트"""
        # 좌측 이미지
        self.axs[0].clear()
        self.axs[0].imshow(color_left[:, :, [2, 1, 0]])
        self.axs[0].set_title("좌측 이미지")
        
        # 우측 이미지
        self.axs[1].clear()
        self.axs[1].imshow(color_right[:, :, [2, 1, 0]])
        self.axs[1].set_title("우측 이미지")
        
        # 체스보드 감지 상태
        self.axs[2].clear()
        if chessboard_detected:
            self.axs[2].text(0.5, 0.5, "체스보드 감지됨!", 
                            ha='center', va='center', transform=self.axs[2].transAxes,
                            fontsize=20, color='green')
        else:
            self.axs[2].text(0.5, 0.5, "체스보드 미감지", 
                            ha='center', va='center', transform=self.axs[2].transAxes,
                            fontsize=20, color='red')
        self.axs[2].set_title("체스보드 상태")
        
        # 전체 제목에 저장된 이미지 수 표시
        self.fig.suptitle(f"실시간 스테레오 카메라 보정 | 저장된 이미지: {self.save_counter}장", fontsize=14)
        
        plt.tight_layout()
        plt.pause(0.001)
        
    def perform_calibration(self):
        """보정 수행"""
        print("\n" + "="*60)
        print("스테레오 카메라 보정 시작")
        print("="*60)
        
        # 저장된 이미지 파일들을 찾기
        left_image_files = sorted(glob.glob(os.path.join(self.save_dir, "left_*.png")))
        right_image_files = sorted(glob.glob(os.path.join(self.save_dir, "right_*.png")))
        
        print(f"찾은 이미지 파일 수: {len(left_image_files)}")
        if len(left_image_files) == 0:
            print("저장된 이미지가 없습니다. 먼저 'c' 키로 이미지를 저장해주세요.")
            return False
            
        # 체스보드가 감지된 이미지들만 수집
        valid_images = []
        for i, (left_path, right_path) in enumerate(zip(left_image_files, right_image_files)):
            color_left = cv.imread(left_path)
            color_right = cv.imread(right_path)
            
            if color_left is None or color_right is None:
                continue
                
            gray_left = cv.cvtColor(color_left, cv.COLOR_BGR2GRAY)
            gray_right = cv.cvtColor(color_right, cv.COLOR_BGR2GRAY)
            
            ret_left, left_object_pts = cv.findChessboardCorners(
                gray_left, self.checkerboard_size, None
            )
            ret_right, right_object_pts = cv.findChessboardCorners(
                gray_right, self.checkerboard_size, None
            )
            
            if ret_left and ret_right:
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                left_object_pts = cv.cornerSubPix(
                    gray_left, left_object_pts, (11, 11), (-1, -1), criteria
                )
                right_object_pts = cv.cornerSubPix(
                    gray_right, right_object_pts, (11, 11), (-1, -1), criteria
                )
                
                valid_images.append({
                    'left_path': left_path,
                    'right_path': right_path,
                    'left_pts': left_object_pts,
                    'right_pts': right_object_pts
                })
                
        print(f"체스보드가 감지된 유효한 이미지: {len(valid_images)}장")
        
        if len(valid_images) < 5:
            print("보정에 필요한 최소 이미지 수(5장)가 부족합니다.")
            return False
            
        # 보정 데이터 준비
        target_object_pts_list = [self.target_object_pts] * len(valid_images)
        left_object_pts_list = [img['left_pts'] for img in valid_images]
        right_object_pts_list = [img['right_pts'] for img in valid_images]
        
        print(f"총 {len(valid_images)}장의 이미지로 보정을 시작합니다.")
        
        # 개별 카메라 보정
        rms1, K1, D1, _, _ = cv.calibrateCamera(
            target_object_pts_list, left_object_pts_list, self.image_size, None, None
        )
        rms2, K2, D2, _, _ = cv.calibrateCamera(
            target_object_pts_list, right_object_pts_list, self.image_size, None, None
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
            self.image_size,
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
            self.image_size,
            R,
            T,
            flags=cv.CALIB_ZERO_DISPARITY,
            alpha=0,  # 여백 제거
        )
        
        # 리매핑 맵 생성
        left_map1, left_map2 = cv.initUndistortRectifyMap(K1, D1, R1, P1, self.image_size, cv.CV_32F)
        right_map1, right_map2 = cv.initUndistortRectifyMap(
            K2, D2, R2, P2, self.image_size, cv.CV_32F
        )
        
        # 결과 저장
        calibration_dir = os.path.join(self.My.GetSrcFilePath(""), "calibration_data")
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
        
        # 보정 결과를 인스턴스 변수로 저장
        self.calibration_data = {
            'R': R, 'T': T, 'K1': K1, 'K2': K2, 'D1': D1, 'D2': D2,
            'R1': R1, 'R2': R2, 'P1': P1, 'P2': P2,
            'left_map1': left_map1, 'left_map2': left_map2,
            'right_map1': right_map1, 'right_map2': right_map2
        }
        
        self.calibration_completed = True
        return True
        
    def show_calibration_results(self):
        """보정 결과 확인"""
        if not self.calibration_completed:
            print("먼저 보정을 완료해주세요 ('b' 키 사용).")
            return
            
        print("\n" + "="*60)
        print("보정 결과 확인")
        print("="*60)
        
        # 저장된 이미지 파일들을 찾기
        left_image_files = sorted(glob.glob(os.path.join(self.save_dir, "left_*.png")))
        right_image_files = sorted(glob.glob(os.path.join(self.save_dir, "right_*.png")))
        
        if len(left_image_files) == 0:
            print("저장된 이미지가 없습니다.")
            return
            
        # 보정 결과 확인용 GUI
        fig_result = plt.figure(figsize=(15, 5))
        key_pressed = None
        
        def on_key_result(event):
            nonlocal key_pressed
            key_pressed = event.key
            
        fig_result.canvas.mpl_connect('key_press_event', on_key_result)
        
        current_idx = 0
        print("단축키: ←/→ (이전/다음), q (종료)")
        
        while True:
            if current_idx >= len(left_image_files):
                current_idx = 0
                
            left_image_path = left_image_files[current_idx]
            right_image_path = right_image_files[current_idx]
            
            color_left = cv.imread(left_image_path)
            color_right = cv.imread(right_image_path)
            
            if color_left is not None and color_right is not None:
                # 보정된 이미지 생성
                rect_left = cv.remap(color_left, self.calibration_data['left_map1'], 
                                   self.calibration_data['left_map2'], cv.INTER_LINEAR)
                rect_right = cv.remap(color_right, self.calibration_data['right_map1'], 
                                    self.calibration_data['right_map2'], cv.INTER_LINEAR)
                
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
                
                axs_result[0].imshow(rect_left[:, :, [2, 1, 0]])
                axs_result[0].set_title(f"보정 후 - 좌측 ({current_idx+1}/{len(left_image_files)})")
                axs_result[1].imshow(rect_right[:, :, [2, 1, 0]])
                axs_result[1].set_title(f"보정 후 - 우측 ({current_idx+1}/{len(left_image_files)})")
                
                depth_map = axs_result[2].imshow(disparity, cmap='plasma')
                axs_result[2].set_title(f"Depth Map ({current_idx+1}/{len(left_image_files)})")
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
                    current_idx = min(len(left_image_files) - 1, current_idx + 1)
                elif key_pressed == 'q':
                    break
                else:
                    print(f"알 수 없는 키: {key_pressed}")
                    
        plt.close(fig_result)
        print("보정 결과 확인 완료!")
        
    def handle_key_events(self):
        """키보드 이벤트 처리"""
        if self.key_pressed == 'c':
            print("현재 이미지를 저장합니다...")
            return 'save'
        elif self.key_pressed == 'b':
            print("보정을 시작합니다...")
            return 'calibrate'
        elif self.key_pressed == 'v':
            print("보정 결과를 확인합니다...")
            return 'verify'
        elif self.key_pressed == 'q':
            print("프로그램을 종료합니다.")
            return 'quit'
            
        self.key_pressed = None
        return None
        
    def run(self):
        """메인 루프"""
        print("실시간 스테레오 카메라 보정을 시작합니다...")
        
        while True:
            # ZMQ 메시지 수신
            msg_parts = None
            while True:
                try:
                    socks = dict(self.poller.poll(timeout=5))
                    if self.sub_socket in socks:
                        msg_parts = self.sub_socket.recv_multipart(flags=zmq.NOBLOCK)
                    else:
                        break
                except zmq.Again:
                    break
                    
            if not msg_parts or len(msg_parts) < 3:
                continue
                
            # 이미지 디코딩
            color_left = cv.imdecode(np.frombuffer(msg_parts[1], dtype=np.uint8), cv.IMREAD_COLOR)
            color_right = cv.imdecode(np.frombuffer(msg_parts[2], dtype=np.uint8), cv.IMREAD_COLOR)
            
            if color_left is None or color_right is None:
                continue
                
            # 체스보드 감지
            chessboard_detected, left_with_corners, right_with_corners, left_pts, right_pts = self.detect_chessboard(color_left, color_right)
            
            # 화면 업데이트
            if chessboard_detected:
                self.update_display(left_with_corners, right_with_corners, True)
            else:
                self.update_display(color_left, color_right, False)
            
            # 키보드 이벤트 처리
            action = self.handle_key_events()
            
            if action == 'save':
                if chessboard_detected:
                    self.save_current_images(color_left, color_right)
                else:
                    print("체스보드가 감지되지 않아 저장하지 않습니다.")
            elif action == 'calibrate':
                self.perform_calibration()
            elif action == 'verify':
                self.show_calibration_results()
            elif action == 'quit':
                break
                
        # 정리
        self.sub_socket.close()
        self.context.term()
        plt.close()
        print(f"\n총 {self.save_counter}장의 이미지가 저장되었습니다.")
        print("\n프로그램이 종료되었습니다.")

if __name__ == "__main__":
    app = StereoRealtimeCalibration()
    app.run() 