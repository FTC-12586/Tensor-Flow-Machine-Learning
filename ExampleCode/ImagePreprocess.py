import cv2
import numpy as np
import os


# Assignment #3, Problem 1
# Solution Manual
# Rod Dockter - ME 5286

# Preprocess images
class ImagePreprocess():
    def __init__(self, dim, colorspace):
        self.dim = dim
        self.colorspace = colorspace

        self.range_x = 20
        self.range_y = 20

    def preprocess(self, image_file):
        # Standard Operations
        image = self._load_image(image_file)
        if (image is None):
            raise ValueError("Image file not found {}".format(image_file))
        image = self._resize(image)
        image = self._colorspace(image)
        image = self._scale_pixels(image)
        return image

    def augment(self, image):
        # Augmentations
        image = self._random_brightness(image)
        image = self._random_flip(image)
        image = self._random_translate(image)
        return image

    def _load_image(self, image_file):
        # Load images from a file
        image = cv2.imread(image_file)
        return image

    def _resize(self, image):
        # resize to specified dimensions
        return cv2.resize(image, (self.dim[1], self.dim[0]), interpolation=cv2.INTER_AREA)

    def _colorspace(self, image):
        # convert to specified color space
        if (self.colorspace == 'HSV'):
            return cv2.cvtcolor(image, cv2.COLOR_BGR2HSV)
        elif (self.colorspace == 'GRAY'):
            return cv2.cvtcolor(image, cv2.COLOR_BGR2GRAY)
        elif (self.colorspace == 'RGB'):
            return cv2.cvtcolor(image, cv2.COLOR_BGR2RGB)
        else:
            return image

    def _scale_pixels(self, image):
        # 0-255 -> -1-1
        image = image.astype(np.float32)
        image = (image - 127.5) / 127.5
        return image

    def _random_flip(self, image):
        # Randomly flip the image left <-> right,
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
        return image

    def _random_translate(self, image):
        # Randomly shift the image vertically and horizontally (translation).
        trans_x = self.range_x * (np.random.rand() - 0.5)
        trans_y = self.range_y * (np.random.rand() - 0.5)
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = image.shape[:2]
        image = cv2.warpAffine(image, trans_m, (width, height))
        return image

    def _random_brightness(self, image):
        # Randomly adjust brightness of the image.
        ratio = 1.0 + 0.3 * (np.random.rand() - 0.5)
        image = image * ratio
        return image
