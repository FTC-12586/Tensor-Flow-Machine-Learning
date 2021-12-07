import copy

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
    new_bbox_y1 = (bordersize + (ul[1] * newHeight)) / wantedHeight
    new_bbox_x2 = br[0]
    new_bbox_y2 = (bordersize + (br[1] * newHeight)) / wantedHeight

    return resized, np.array([new_bbox_x1, new_bbox_y1], dtype=float), np.array([new_bbox_x2, new_bbox_y2], dtype=float)


def resize_bbox(Img: np.ndarray, dim_out, ul, br):
    wantedWidth = dim_out[0]
    wantedHeight = dim_out[1]

    width, height = cv_size(Img)

    bordersize = int((wantedHeight - height) / 2)

    new_bbox_x1 = ul[0]
    new_bbox_y1 = (bordersize + (ul[1] * height)) / wantedHeight
    new_bbox_x2 = br[0]
    new_bbox_y2 = (bordersize + (br[1] * height)) / wantedHeight

    return np.array([new_bbox_x1, new_bbox_y1], dtype=float), np.array([new_bbox_x2, new_bbox_y2], dtype=float)


def showBBOX(Img, ul, br):
    Img2 = copy.deepcopy(Img)
    Img2 = cv2.rectangle(Img2, ul, br, (255, 0, 0), 2)
    cv2.imshow("image", Img2)
    cv2.waitKey(5000)


def decimal_bbox_to_abs_bbox(Img, ul, br):
    width, height = cv_size(Img)
    abs_ul = np.array([0, 0], dtype=int)
    abs_ul[0] = int(width * ul[0])
    abs_ul[1] = int(height * ul[1])

    abs_br = np.array([0, 0], dtype=int)
    abs_br[0] = int(width * br[0])
    abs_br[1] = int(height * br[1])

    return Img, abs_ul, abs_br


def abs_bbox_to_decimal_bbox(Img, ul, br):
    width, height = cv_size(Img)
    dec_ul = np.array([0, 0], dtype=float)
    dec_ul[0] = ul[0] / width
    dec_ul[1] = ul[1] / height

    dec_br = np.array([0, 0], dtype=float)
    dec_br[0] = br[0] / width
    dec_br[1] = br[1] / height

    return Img, dec_ul, dec_br
