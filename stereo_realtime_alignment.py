import cv2 as cv
import numpy as np
import zmq
import math
import tifffile
from matplotlib import pyplot as plt
from MyCommon import MyCommon
import os

class StereoRealtimeAlignment:
    def __init__(self):
        # MyCommon 초기화
        self.My = MyCommon("E:/res/STEREO_VIDEO", "calibration_data", "TEST")
        
        # 보정 데이터 로드
        self.T = tifffile.imread(self.My.GetSrcFilePath("T.tif"))
        self.P1 = tifffile.imread(self.My.GetSrcFilePath("P1.tif"))
        self.left_map1 = tifffile.imread(self.My.GetSrcFilePath("left_map1.tif"))
        self.left_map2 = tifffile.imread(self.My.GetSrcFilePath("left_map2.tif"))
        self.right_map1 = tifffile.imread(self.My.GetSrcFilePath("right_map1.tif"))
        self.right_map2 = tifffile.imread(self.My.GetSrcFilePath("right_map2.tif"))
        
        # 카메라 파라미터
        self.image_size = (960, 720)
        self.focal_length = float(self.P1[0, 0])
        self.baseline_mm = float(abs(self.T[0]))
        
        print(f"카메라 정보:")
        print(f"  Focal Length: {self.focal_length:.2f} pixels")
        print(f"  Baseline: {self.baseline_mm:.2f} mm")
        
        # ZMQ 설정
        self.context = zmq.Context()
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.setsockopt(zmq.RCVHWM, 1)
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 10)
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "STATE_CAMERA_SENSOR")
        self.sub_socket.connect("tcp://192.168.135.32:45000")
        
        self.poller = zmq.Poller()
        self.poller.register(self.sub_socket, zmq.POLLIN)
        
        # 정합도 확인용 변수
        self.current_distance_mm = 200  # 초기 거리 20cm
        self.key_pressed = None
        
        # GUI 설정
        self.setup_gui()
        
    def setup_gui(self):
        """GUI 초기화"""
        self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # 서브플롯 제목 설정
        self.axs[0].set_title("좌측 이미지")
        self.axs[1].set_title("우측 이미지")
        self.axs[2].set_title("정합도 확인")
        
        # 컬러바 설정
        self.im = self.axs[2].imshow(np.zeros((100, 100)))
        self.cbar = self.fig.colorbar(self.im, ax=self.axs[2], fraction=0.046, pad=0.04)
        
        print("\n" + "="*60)
        print("실시간 스테레오 정합도 확인 프로그램")
        print("="*60)
        print("단축키 설명:")
        print("  ↑/↓ : 거리 조정 (10mm 단위)")
        print("  a/s : 거리 조정 (50mm 단위)")
        print("  z/x : 거리 조정 (100mm 단위)")
        print("  r    : 거리 초기화")
        print("  q    : 프로그램 종료")
        print("="*60)
        
    def on_key(self, event):
        """키보드 이벤트 처리"""
        self.key_pressed = event.key
        
    def calculate_disparity(self, distance_mm):
        """거리에 따른 시차 계산"""
        if distance_mm <= 0:
            return 0
        # 스테레오 시차 공식: disparity = (focal_length * baseline) / distance
        disparity_pixels = (self.focal_length * self.baseline_mm) / distance_mm
        return disparity_pixels
        
    def shift_image_for_distance(self, image, disparity_pixels):
        """거리에 따라 이미지를 이동"""
        height, width = image.shape[:2]
        # 우측 이미지를 좌측으로 이동 (시차만큼)
        shift_matrix = np.float32([[1, 0, disparity_pixels], [0, 1, 0]])
        shifted_image = cv.warpAffine(image, shift_matrix, (width, height))
        return shifted_image
        
    def process_frame(self, left_rect, right_rect):
        """프레임 처리 및 정합도 계산"""
        # 현재 거리에 따른 시차 계산
        disparity_pixels = self.calculate_disparity(self.current_distance_mm)
        
        # 거리에 따라 우측 이미지 이동
        shifted_right = self.shift_image_for_distance(right_rect, disparity_pixels)
        
        # 이미지 겹치기 (좌측은 빨간색, 우측은 청록색)
        overlay = np.zeros_like(left_rect)
        overlay[:, :, 2] = left_rect[:, :, 2]  # 빨간색 채널 (좌측)
        overlay[:, :, 0] = shifted_right[:, :, 0]  # 파란색 채널 (우측)
        overlay[:, :, 1] = shifted_right[:, :, 1]  # 초록색 채널 (우측)
        
        # 정합도 계산
        diff = cv.absdiff(left_rect, shifted_right)
        match_score = 255 - np.mean(diff)
        match_percentage = (match_score / 255) * 100
        
        return overlay, match_percentage, disparity_pixels
        
    def update_display(self, left_rect, right_rect, overlay, match_percentage, disparity_pixels):
        """화면 업데이트"""
        # 좌측 이미지
        self.axs[0].clear()
        self.axs[0].imshow(left_rect[:, :, [2, 1, 0]])
        self.axs[0].set_title("좌측 이미지")
        
        # 우측 이미지
        self.axs[1].clear()
        self.axs[1].imshow(right_rect[:, :, [2, 1, 0]])
        self.axs[1].set_title("우측 이미지")
        
        # 정합도 확인 이미지
        self.axs[2].clear()
        self.axs[2].imshow(overlay[:, :, [2, 1, 0]])
        self.axs[2].set_title(f"정합도 확인\n거리: {self.current_distance_mm}mm | 시차: {disparity_pixels:.1f}px")
        
        # 전체 제목에 정합도 정보 추가
        self.fig.suptitle(f"실시간 스테레오 정합도 확인 | 정합도: {match_percentage:.1f}%", fontsize=14)
        
        plt.tight_layout()
        plt.pause(0.001)
        
    def handle_key_events(self):
        """키보드 이벤트 처리"""
        if self.key_pressed == 'up':
            self.current_distance_mm += 10
            print(f"거리 증가: {self.current_distance_mm}mm (+10mm)")
        elif self.key_pressed == 'down':
            self.current_distance_mm = max(10, self.current_distance_mm - 10)
            print(f"거리 감소: {self.current_distance_mm}mm (-10mm)")
        elif self.key_pressed == 'ctrl+up':
            self.current_distance_mm += 50
            print(f"거리 증가: {self.current_distance_mm}mm (+50mm)")
        elif self.key_pressed == 'ctrl+down':
            self.current_distance_mm = max(10, self.current_distance_mm - 50)
            print(f"거리 감소: {self.current_distance_mm}mm (-50mm)")
        elif self.key_pressed == 'shift+up':
            self.current_distance_mm += 100
            print(f"거리 증가: {self.current_distance_mm}mm (+100mm)")
        elif self.key_pressed == 'shift+down':
            self.current_distance_mm = max(10, self.current_distance_mm - 100)
            print(f"거리 감소: {self.current_distance_mm}mm (-100mm)")
        elif self.key_pressed == 'r':
            self.current_distance_mm = 200
            print(f"거리 초기화: {self.current_distance_mm}mm")
        elif self.key_pressed == 'q':
            print("프로그램을 종료합니다.")
            return False
            
        self.key_pressed = None
        return True
        
    def run(self):
        """메인 루프"""
        print("실시간 스테레오 정합도 확인을 시작합니다...")
        
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
                
            # 보정된 이미지 생성
            left_rect = cv.remap(color_left, self.left_map1, self.left_map2, cv.INTER_LINEAR)
            right_rect = cv.remap(color_right, self.right_map1, self.right_map2, cv.INTER_LINEAR)
            
            # 프레임 처리
            overlay, match_percentage, disparity_pixels = self.process_frame(left_rect, right_rect)
            
            # 화면 업데이트
            self.update_display(left_rect, right_rect, overlay, match_percentage, disparity_pixels)
            
            # 키보드 이벤트 처리
            if not self.handle_key_events():
                break
                
        # 정리
        self.sub_socket.close()
        self.context.term()
        plt.close()
        print("\n프로그램이 종료되었습니다.")

if __name__ == "__main__":
    app = StereoRealtimeAlignment()
    app.run() 