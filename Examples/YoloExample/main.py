from yolo_network import YOLOV1
from data_generator import DataGenerator
from utils import DecodeYoloOutput, DrawYoloOutput

import cv2
import numpy as np

num_classes = 1

## YOLO Model
yolo = YOLOV1()
model = yolo.create_model(num_classes=num_classes)
yolo.display_model()

## Data Generators
trainrecord = "/home/rodney/python_ws/tensorflow_frc/dataset/train_dataset.record-00000-00001"
evalrecord = "/home/rodney/python_ws/tensorflow_frc/dataset/eval_dataset.record-00000-00001"
labelfile = "/home/rodney/python_ws/tensorflow_frc/dataset/label.pbtxt"
batch_size = 20

train_generator = DataGenerator(trainrecord, labelfile, image_dims=[448, 448, 3], num_classes=num_classes,
                                batch_size=batch_size)
eval_generator = DataGenerator(evalrecord, labelfile, image_dims=[448, 448, 3], num_classes=num_classes,
                               batch_size=batch_size)

x_train, y_train = train_generator.__getitem__(0)
print("Input Shape: {}".format(x_train.shape))
print("Output Shape: {}".format(y_train.shape))

sample = x_train[0]
print(sample.shape)

estimate = yolo.inference(sample)
print(estimate.shape)

boxes = DecodeYoloOutput(estimate, threshold=0.6)
print(boxes)

## Train the model
yolo.train(train_generator, eval_generator, epochs=135, weights_file='weight.hdf5')

x_val, y_val = eval_generator.__getitem__(0)
sample = x_val[0]
estimate = yolo.inference(sample[None])

# get yolo decode
boxes = DecodeYoloOutput(estimate, threshold=0.6)
print(boxes)

# draw
result = DrawYoloOutput(sample, estimate)

cv2.imshow("output", result)
cv2.waitKey(0)
