import unittest, os
from OurCode import *
from data_generator import DataGenerator
import numpy as np
import cv2


class ImageManipulation(unittest.TestCase):
    def testSizes(self):
        img = cv2.imread("download.jfif")
        img1, ul, br = ResizeFill(img, [448, 448, 3], (0, 0), (0, 0))
        img2, ul, br = DataGenerator.ResizeCrop(img, [448, 448, 3], (0, 0), (0, 0))

        self.assertEqual(img1.size, img2.size)

    def ResizeFillTestReturnedDataTypes(self):
        img = cv2.imread("download.jfif")
        data = ResizeFill(img, [448, 448, 3], (0, 0), (0, 0))
        data2 = DataGenerator.ResizeCrop(img, [448, 448, 3], (0, 0), (0, 0))

        self.assertEqual(len(data), len(data2))
        self.assertEqual(len(data), 3)

        self.assertEqual(type(data[-1]), type(data2[-1]))
        self.assertEqual(type(data[-2]), type(data2[-2]))

    def ResizeFillTestReturnedRange(self):
        img = cv2.imread("download.jfif")
        points = np.linspace(0, 1, 480, endpoint=False)
        for k in range(0, len(points)):
            i = points[k]
            img1, ul, br = ResizeFill(img, [448, 448, 3], (i, i), (i, i))
            self.assertEqual((ul[0] <= 1.0), True)
            self.assertEqual((ul[0] >= 0), True)
            self.assertEqual((ul[1] <= 1.0), True)
            self.assertEqual((ul[1] >= 0), True)

            self.assertEqual((br[0] <= 1.0), True)
            self.assertEqual((br[0] >= 0), True)
            self.assertEqual((br[1] <= 1.0), True)
            self.assertEqual((br[1] >= 0), True)

    def testResizeFill(self):
        img = cv2.imread("download.jfif")
        ul = (.5, .5)
        br = (.6, .6)
        img, ul2, br2 = decimal_bbox_to_abs_bbox(img,ul,br)
        img2, ul, br = ResizeFill(img, [448, 448, 3], ul, br)
        showBBOX(img2, ul2, br2)
        width, height = cv_size(img2)
        self.assertEqual(width,448)
        self.assertEqual(height, 448)


if __name__ == '__main__':
    unittest.main()
