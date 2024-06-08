import numpy as np


def get_center_of_bounding_box(bounding_box) -> tuple[int, int]:
    x1, y1, x2, y2 = bounding_box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bounding_box_width(bounding_box):
    x1, y1, x2, y2 = bounding_box
    return int(x2 - x1)


def measure_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
