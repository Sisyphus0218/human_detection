def calculate_iou(
    first_bbox: tuple[int, int, int, int],
    second_bbox: tuple[int, int, int, int],
) -> float:
    """Calculate intersection over union for two xyxy boxes."""
    first_x1, first_y1, first_x2, first_y2 = first_bbox
    second_x1, second_y1, second_x2, second_y2 = second_bbox

    intersection_width = max(
        0,
        min(first_x2, second_x2) - max(first_x1, second_x1),
    )
    intersection_height = max(
        0,
        min(first_y2, second_y2) - max(first_y1, second_y1),
    )
    intersection_area = intersection_width * intersection_height

    first_area = max(0, first_x2 - first_x1) * max(0, first_y2 - first_y1)
    second_area = max(0, second_x2 - second_x1) * max(0, second_y2 - second_y1)
    union_area = first_area + second_area - intersection_area
    if union_area <= 0:
        return 0.0

    return intersection_area / union_area
