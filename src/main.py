import cv2
import os

from data_generator import DataGenerator
from utils import DecodeYoloOutput, DrawYoloOutput
from yolo_network import YOLOV1


def main():
    num_classes = 1

    ## YOLO Model
    yolo = YOLOV1()
    model = yolo.create_model(num_classes=num_classes)
    yolo.display_model()

    ## Data Generators
    trainrecord = os.path.abspath(os.path.join("..", "datasets", "Dataset1", "train_dataset.record-00000-00001"))
    evalrecord = os.path.abspath(os.path.join("..", "datasets", "Dataset1", "eval_dataset.record-00000-00001"))
    labelfile = os.path.abspath(os.path.join("..", "datasets", "Dataset1", "label.pbtxt"))
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
    yolo.train(train_generator, eval_generator, epochs=1, weights_file='weight.hdf5')

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


if __name__ == "__main__":
    main()
