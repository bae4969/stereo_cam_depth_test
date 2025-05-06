import cv2 as cv


def GetVideoCapcture(filepath:str):
    video_cap = cv.VideoCapture(filepath)  # <-- 파일명 수정해줘
    if not video_cap.isOpened():
        return None
    
    return video_cap


def GetMaxFrameIndex(video_cap: cv.VideoCapture):
    return int(video_cap.get(cv.CAP_PROP_FRAME_COUNT))


def GetFrameWithTimeStamp(video_cap: cv.VideoCapture, timestamp_sec: int):
    fps = video_cap.get(cv.CAP_PROP_FPS)
    total_frames = int(video_cap.get(cv.CAP_PROP_FRAME_COUNT))
    target_frame = int(timestamp_sec * fps)

    if total_frames < target_frame:
        return False, None, None

    video_cap.set(cv.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = video_cap.read()
    if not ret:
        return False, None, None
    
    h, w, _ = frame.shape
    half_w = w // 2
    imgL = frame[:, :half_w]
    imgR = frame[:, half_w:]

    return True, imgL, imgR


def GetFrameWithIndex(video_cap: cv.VideoCapture, frame_index: int):
    total_frames = int(video_cap.get(cv.CAP_PROP_FRAME_COUNT))
    target_frame = frame_index

    if total_frames < target_frame:
        return False, None, None

    video_cap.set(cv.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = video_cap.read()
    if not ret:
        return False, None, None
    
    h, w, _ = frame.shape
    half_w = w // 2
    imgL = frame[:, :half_w]
    imgR = frame[:, half_w:]

    return True, imgL, imgR
