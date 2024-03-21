import cv2
from ultralytics import YOLO

# YOLO 실행
model = YOLO("yolov8n.pt")

# 비디오 파일 경로
video_path = "bb_1_000105_vehicle_48_131.mp4"
cap = cv2.VideoCapture(video_path)

# 비디오 프레임 실행
while cap.isOpened():
    # 비디오에서 프레임 읽기
    success, frame = cap.read()

    if success:

        # 읽은 프레임에서 YOLO 실행
        results = model(frame)
        
        # 탐지한 객체를 프레임에 입력하여 새로운 프레임을 생성
        annotated_frame = results[0].plot()
        
        # 생성한 프레임 출력
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # 'q' 버튼을 입력하면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 비디오 프레임이 끝나면 종료
        break

# 프로그램을 종료하고 창 닫기
cap.release()
cv2.destroyAllWindows()