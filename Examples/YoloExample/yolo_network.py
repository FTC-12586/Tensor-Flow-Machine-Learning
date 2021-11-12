import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Lambda, InputLayer
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

from learning_rate import LearningRateScheduler
from utils import YOLO_Loss
from yolo_reshape import Yolo_Reshape


class YOLOV1:
    def __init__(self):
        self.grid_w = 7
        self.grid_h = 7
        self.cell_w = 64
        self.cell_h = 64
        self.img_w = self.grid_w * self.cell_w  # 448
        self.img_h = self.grid_h * self.cell_h  # 448

        self.yolo_loss = None

        self.LR_SCHEDULE = [
            # (epoch to start, learning rate) tuples
            (0, 0.01),
            (75, 0.001),
            (105, 0.0001),
        ]

    def linear_transform(self, inputs):
        inputs = tf.reverse(inputs, [-1]) - tf.constant([103.939, 116.779, 123.68])
        return inputs / 255.0

    def display_model(self):
        self.model.summary()

    def lr_schedule(self, epoch, lr):
        """Helper function to retrieve the scheduled learning rate based on epoch."""
        if epoch < self.LR_SCHEDULE[0][0] or epoch > self.LR_SCHEDULE[-1][0]:
            return lr
        for i in range(len(self.LR_SCHEDULE)):
            if epoch == self.LR_SCHEDULE[i][0]:
                return self.LR_SCHEDULE[i][1]
        return lr

    def inference(self, input):
        return self.model.predict(input)

    def train(self, train_generator, eval_generator, epochs=135, weights_file='weight.hdf5'):
        # Checkpoint
        mcp_save = ModelCheckpoint(weights_file, save_best_only=True, monitor='val_loss', mode='min')

        self.model.fit(x=train_generator,
                       steps_per_epoch=train_generator.get_steps_per_epoch(),
                       epochs=epochs,
                       verbose=1,
                       workers=4,
                       validation_data=eval_generator,
                       validation_steps=eval_generator.get_steps_per_epoch(),
                       callbacks=[
                           LearningRateScheduler(self.lr_schedule),
                           mcp_save
                       ])

    def create_model(self, num_classes=20, nb_boxes=1):
        self.num_classes = num_classes
        self.nb_boxes = nb_boxes

        lrelu = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.model = Sequential()

        self.model.add(InputLayer(input_shape=(self.img_h, self.img_w, 3)))
        self.model.add(Lambda(self.linear_transform))

        self.model.add(Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation=lrelu,
                              kernel_regularizer=l2(5e-4)))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        self.model.add(
            Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        self.model.add(
            Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        self.model.add(
            Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        self.model.add(
            Conv2D(filters=256, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        self.model.add(
            Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        self.model.add(
            Conv2D(filters=256, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        self.model.add(
            Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        self.model.add(
            Conv2D(filters=256, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        self.model.add(
            Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        self.model.add(
            Conv2D(filters=256, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        self.model.add(
            Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        self.model.add(
            Conv2D(filters=256, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        self.model.add(
            Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        self.model.add(
            Conv2D(filters=512, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        self.model.add(
            Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        self.model.add(
            Conv2D(filters=512, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        self.model.add(
            Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        self.model.add(
            Conv2D(filters=512, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        self.model.add(
            Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        self.model.add(
            Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
        self.model.add(Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), padding='same'))

        self.model.add(Conv2D(filters=1024, kernel_size=(3, 3), activation=lrelu, kernel_regularizer=l2(5e-4)))
        self.model.add(Conv2D(filters=1024, kernel_size=(3, 3), activation=lrelu, kernel_regularizer=l2(5e-4)))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Dense(1024))
        self.model.add(Dropout(0.5))

        # 7x7xn output
        outlen = (self.nb_boxes * 5) + num_classes  # 20 + 2*5 = 30
        denselen = self.grid_w * self.grid_h * outlen  # 7x7x30 = 1470

        self.model.add(Dense(denselen, activation='sigmoid'))
        self.model.add(Yolo_Reshape(target_shape=(self.grid_w, self.grid_h, outlen), num_classes=self.num_classes,
                                    nb_boxes=self.nb_boxes))

        self.yolo_loss = YOLO_Loss(num_classes)
        self.model.compile(loss=self.yolo_loss.yolo_loss, optimizer='adam')

        return self.model
