import cv2
import numpy as np


def extract_front_edge(mask):
    """
    Extract the approximately horizontal lower/front edge
    from a binary pallet_front mask.

    Returns:
        left_point:  (x, y)
        right_point: (x, y)
    """

    ys, xs = np.where(mask > 0)

    if len(xs) < 10:
        return None, None

    # Bottom portion of the detected pallet-front region.
    y_threshold = np.percentile(ys, 90)

    bottom = np.column_stack(
        np.where(
            mask &
            (np.indices(mask.shape)[0] >= y_threshold)
        )
    )

    if len(bottom) < 2:
        return None, None

    # bottom = [y, x]
    left_idx = np.argmin(bottom[:, 1])
    right_idx = np.argmax(bottom[:, 1])

    left = (
        int(bottom[left_idx, 1]),
        int(bottom[left_idx, 0])
    )

    right = (
        int(bottom[right_idx, 1]),
        int(bottom[right_idx, 0])
    )

    return left, right


def draw_geometry(image, left, right):

    output = image.copy()

    if left is None or right is None:
        return output

    cv2.circle(
        output,
        left,
        8,
        (0, 255, 0),
        -1
    )

    cv2.circle(
        output,
        right,
        8,
        (0, 0, 255),
        -1
    )

    cv2.line(
        output,
        left,
        right,
        (255, 0, 0),
        3
    )

    return output