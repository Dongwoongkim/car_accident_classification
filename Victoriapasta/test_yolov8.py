import cv2
import os
from ultralytics import YOLO

# YOLO 모델 초기화
model = YOLO("yolov8n.pt")

# 분석할 비디오 파일 경로 리스트
global video_paths
video_paths = ["yolotest1.mp4", "yolotest2.mp4", "yolotest3.mp4"]

def explore_directory(directory):
    # 디렉토리 내의 모든 파일과 디렉토리에 대해 순회
    for root, dirs, files in os.walk(directory):
        # 디렉토리 내의 파일들 출력
        for file in files:
            # 파일의 경로 출력
            file_path = os.path.join(root, file)
            video_paths.append(file_path)

        # 디렉토리 내의 하위 디렉토리들에 대해 재귀적으로 탐색
        for dir_name in dirs:
            explore_directory(os.path.join(root, dir_name))

explore_directory("d:\Project Datas\Datas\Training\Original Data\TS_T-intersection")
# 각 비디오에 대해 순회
for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)

    # 비디오 프레임 실행
    while cap.isOpened():
        # 비디오에서 프레임 읽기
        success, frame = cap.read()

        if success:
            # 읽은 프레임에서 YOLO 실행
            results = model(frame)
            
            # 탐지한 객체를 프레임에 입력하여 새로운 프레임 생성
            annotated_frame = results[0].plot()
            
            # 생성한 프레임 출력
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # 'q' 버튼을 입력하면 종료
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # 비디오 프레임이 끝나면 종료
            break

    # 비디오 파일을 닫기
    cap.release()

# 모든 창 닫기
cv2.destroyAllWindows()