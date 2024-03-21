import os
import json
import cv2
import matplotlib.pyplot as plt

from ultralytics import YOLO

# YOLO 모델 초기화
model = YOLO("yolov8n.pt")

# 이미지가 있는 디렉토리 경로 설정
image_directory = "data/image/bb_1_000101_vehicle_29_113/"

# 결과를 저장할 디렉토리 생성
result_directory = "data/results/"

# 해당 디렉토리 존재하지 않는 경우, 디렉토리 생성
os.makedirs(result_directory, exist_ok=True)

# 레이블 데이터에서 좌상단으로 매핑된 x,y값을 통해 바운딩 박스의 중앙 x값, 중앙 y값 계산하는 함수
def calculate_center(bbox):
    x, y, w, h = bbox
    center_x = x + w / 2
    center_y = y + h / 2
    return center_x, center_y

# 바운딩 박스 간 일치율을 계산하는 함수
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 바운딩 박스의 좌표를 기반으로 직사각형의 영역을 계산합니다.
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    # 직사각형의 영역을 계산합니다.
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # 각 바운딩 박스의 영역을 계산합니다.
    area1 = w1 * h1
    area2 = w2 * h2

    # 교차하는 영역을 제외한 두 바운딩 박스의 합집합 영역을 계산합니다.
    union_area = area1 + area2 - intersection_area

    # 일치율을 계산합니다.
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

graph = []

# 이미지 디렉토리 내 모든 이미지에 대해 Yolo 모델 추론 실행
for filename in os.listdir(image_directory):

    if filename.endswith(".png"):  # 이미지 파일만 대상으로 함
        image_path = os.path.join(image_directory, filename)

        # YOLOv8 모델로 이미지 추론 실행
        results = model(image_path)
        result_frame = results[0].plot()

        # 추론 결과 저장
        output_image_path = os.path.join(result_directory, filename[:-4] + "_result.png")
        # cv2.imwrite(output_image_path, result_frame)

        # 객체 정보 출력
        boxes = results[0].boxes.xywh.tolist()
        classes = results[0].boxes.cls.tolist()
        names = results[0].names
        confidences = results[0].boxes.conf.tolist()

        other_boxes = []
        print("Results for:", filename)
        for box, cls, conf in zip(boxes, classes, confidences):
            x, y, w, h = box
            confidence = conf
            detected_class = cls
            name = names[int(cls)]

            other_boxes.append([x, y, w, h])
            print("(x, y, w, h) : ", x, y, w, h)

        # JSON 파일 로드 및 객체 정보 출력
        json_filename = filename[:-4] + ".json"
        json_filepath = os.path.join("data/label/bb_1_000101_vehicle_29_113/", json_filename)

        with open(json_filepath) as file:
            datas = json.load(file)

            objects = datas["objects"]
            for i in objects:
                if i["isObjectA"]:
                    bbox_i = i["bbox"]
                    calculated_center_box = calculate_center(bbox_i)
                    new_box = [calculated_center_box[0], calculated_center_box[1], bbox_i[2], bbox_i[3]]
                    print(new_box)

                    iou_list = []
                    for other_box in other_boxes:
                        iou = calculate_iou(new_box, other_box)
                        iou_list.append(iou)

                    # 내림차순 정렬
                    iou_list.sort(reverse=True)

                    if iou_list:
                        graph.append(iou_list[0])
                        print(iou_list[0])

        print("\n")

# 히스토그램 그리기
plt.hist(graph, bins=20)
plt.xlabel("IoU")
plt.ylabel("Frequency")
plt.title("IoU Distribution")
plt.show()
