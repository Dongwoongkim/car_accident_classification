def calculate_center(bbox):
    x, y, w, h = bbox
    center_x = x + w / 2
    center_y = y + h / 2
    return center_x, center_y


bbox = [100, 100, 200, 200]
print(calculate_center(bbox))

