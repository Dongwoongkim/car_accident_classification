import json
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
image_path = "data/image/bb_1_000101_vehicle_29_113/113_29_001"
png = ".png"
image_file = image_path + png
results = model(image_file)
result_frame = results[0].plot()

boxes = results[0].boxes.xywh.tolist()
classes = results[0].boxes.cls.tolist()
names = results[0].names
confidences = results[0].boxes.conf.tolist()

for box, cls, conf in zip(boxes, classes, confidences):
    x, y, w, h = box
    confidence = conf
    detected_class = cls
    name = names[int(cls)]

    print("(x, y, w, h) : ", x, y, w, h)
    print("confidence : ", confidence)
    print("class : ", detected_class)
    print("name : ", name)

cv2.imwrite(image_path + "_result.png", result_frame)
#cv2.imshow("YOLOv8 Inference", result_frame)

with open("data/label/bb_1_000101_vehicle_29_113/113_29_001.json") as file:
    datas = json.load(file)

    objects = datas["objects"]
    for i in objects:
        print(i["bbox"])