import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import cv2
import os

from ImagePreprocess import ImagePreprocess


# Assignment #3, Problem 4-6
# Solution Manual
# Rod Dockter - ME 5286

class CustomNetwork():
    # Define model
    def __init__(self, num_classes=4, input_shape=(240, 320, 3)):
        # define model
        self.model = keras.models.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax')
        ])

        self.image_preprocess = ImagePreprocess(input_shape, 'BGR')

        # Loss function
        self.loss_fn = keras.losses.CategoricalCrossentropy()

        # Optimizer
        self.opt_fn = keras.optimizers.Adam(learning_rate=0.001)

        # Set backprop Type
        self.model.compile(optimizer=self.opt_fn, loss=self.loss_fn, metrics=['accuracy'])

    def LoadModel(self, stored_model):
        self.model = keras.models.load_model(stored_model)

    def TrainModel(self, train_generator, validate_generator, epochs, preload, stored_model):
        # Checkpoint
        checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=stored_model, save_weights_only=False,
                                                              monitor='val_accuracy', mode='max', save_best_only=True)

        if preload:
            self.model = keras.models.load_model(stored_model)

        # Train model on dataset
        self.model.fit(train_generator, validation_data=validate_generator, epochs=epochs,
                       callbacks=[checkpoint_callback])

    def InferenceModel(self, image_path):
        image = self.image_preprocess.preprocess(image_path)

        # Forward pass
        result = self.model(image[None,]).numpy()
        class_index = np.argmax(result)

        return class_index
