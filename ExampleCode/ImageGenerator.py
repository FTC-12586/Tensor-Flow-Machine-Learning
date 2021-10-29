import numpy as np
import tensorflow.keras as keras
import os
import cv2
from ImagePreprocess import ImagePreprocess


# Assignment #3, Problem 3
# Solution Manual
# Rod Dockter - ME 5286

# Use LoadDataSet to read data folder list

class ImageGenerator(keras.utils.Sequence):
    # Generate Training Data In Batches
    def __init__(self, images, labels, path, n_classes, batch_size=32, dim=(240, 320, 3), augment=True, shuffle=True):
        # Intialize Variables
        self.image_preprocess = ImagePreprocess(dim, 'BGR')
        self.path = path

        self.labels = labels  # Dictionary
        self.images = images  # List
        self.n_classes = n_classes  # int

        self.batch_size = batch_size  # int
        self.dim = dim  # tuple
        self.augment = augment  # bool
        self.shuffle = shuffle  # bool

        self.on_epoch_end()

    def __len__(self):
        # Number of batches per epoch
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[(index * self.batch_size):((index + 1) * self.batch_size)]

        # Find list of IDs
        image_ids_temp = [self.images[k] for k in indexes]

        # Generate data
        X, Y = self.__create_next_batch(image_ids_temp)

        return X, Y

    def on_epoch_end(self):
        # Resample random indexes after each epoch
        self.indexes = np.arange(len(self.images))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __create_next_batch(self, image_ids_temp):
        # Generate next batch of data

        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        Y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, file in enumerate(image_ids_temp):
            # load and process image
            file_path = os.path.join(self.path, file)
            image = self.image_preprocess.preprocess(file_path)

            # augment maybe
            if (self.augment):
                image = self.image_preprocess.augment(image)

            # store image and label
            X[i,] = image
            Y[i] = self.labels[file]

        return X, self.__to_one_hot(Y)

    def __to_one_hot(self, labels):
        one_hot = np.zeros((labels.size, self.n_classes))
        for i, cid in enumerate(labels):
            one_hot[i, cid] = 1
        return one_hot
