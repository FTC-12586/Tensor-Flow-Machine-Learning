import cv2
import numpy as np


def cv_size(img) -> tuple:
    return tuple(img.shape[1::-1])


def ResizeFill(Img: np.ndarray, dim_out, ul, br) -> tuple:
    wantedWidth = dim_out[0]
    wantedHeight = dim_out[1]
    width, height = cv_size(Img)
    m = wantedWidth / width
    width = int(width * m)
    height = int(height * m)
    newHeight = height
    if height % 2 != 0:
        height = height + 1
    resized = cv2.resize(Img, (width, height))

    width, height = cv_size(resized)

    bordersize = int((wantedHeight - height) / 2)

    resized = cv2.copyMakeBorder(
        resized,
        top=bordersize,
        bottom=bordersize,
        left=0,
        right=0,
        borderType=cv2.BORDER_CONSTANT
    )
    # TODO Fix the Bounding Boxes. They are incorrectly being recalculated
    new_bbox_x1 = ul[0]
    new_bbox_y1 = (bordersize + (ul[1]*newHeight))/wantedHeight
    new_bbox_x2 = br[0]
    new_bbox_y2 = (bordersize + (br[1]*newHeight))/wantedHeight

    return resized, np.array([new_bbox_x1, new_bbox_y1], dtype=float), np.array([new_bbox_x2, new_bbox_y2], dtype=float)


def resize_bbox(Img: np.ndarray, dim_out, ul, br):
    # TODO Fix the Bounding Boxes. They are incorrectly being recalculated
    wantedWidth = dim_out[0]
    wantedHeight = dim_out[1]

    width, height = cv_size(Img)


    bordersize = int((wantedHeight - height) / 2)

    new_bbox_x1 = ul[0]
    new_bbox_y1 = (bordersize + (ul[1]*height))/wantedHeight
    new_bbox_x2 = br[0]
    new_bbox_y2 = (bordersize + (br[1]*height))/wantedHeight

    return np.array([new_bbox_x1, new_bbox_y1], dtype=float), np.array([new_bbox_x2, new_bbox_y2], dtype=float)
