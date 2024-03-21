def calculate_iou(bbox1, bbox2):
    # 바운딩 박스의 좌표를 추출합니다.
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

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


print(calculate_iou([1005,528,266,191],[1105,528,266,197]))