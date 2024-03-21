import json
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
image_path = "C:/Users/jbnu/Desktop/ML Project/113_29_105"
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

    print(x, y, w, h)
    print(confidence)
    print(detected_class)
    print(name)

cv2.imwrite(image_path + "_result.png", result_frame)
#cv2.imshow("YOLOv8 Inference", result_frame)

with open("C:/Users/jbnu/Desktop/ML Project/113_29_105.json") as file:
    datas = json.load(file)

    objects = datas["objects"]
    for i in objects:
        print(i["bbox"])