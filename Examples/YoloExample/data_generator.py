from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2

from utils import read_label_map


class DataGenerator(keras.utils.Sequence):

    def __init__(self, recordfile, labelfile, image_dims=[448, 448, 3], num_classes=20, batch_size=20):
        self.recordfile = recordfile
        self.labelfile = labelfile

        self.image_dims = image_dims
        self.output_dims = [7, 7, num_classes + 5]
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.raw_dataset = tf.data.TFRecordDataset(recordfile)
        self.totalsamples = sum(1 for _ in self.raw_dataset)  # num images in record

        self.labels = read_label_map(labelfile)

        if (self.totalsamples < self.batch_size):
            self.batch_size = self.totalsamples
            print("Reducing Batch Size to {}".format(self.batch_size))

        # Each element in the .record dataset has these features
        self.feature_description = {
            'image/source_id': tf.io.FixedLenFeature([], tf.string),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'image/format': tf.io.FixedLenFeature([], tf.string),
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/object/bbox/xmin': tf.io.FixedLenFeature([], tf.float32),
            'image/object/bbox/xmax': tf.io.FixedLenFeature([], tf.float32),
            'image/object/bbox/ymin': tf.io.FixedLenFeature([], tf.float32),
            'image/object/bbox/ymax': tf.io.FixedLenFeature([], tf.float32),
            'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
            'image/object/class/text': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
        }

    def __len__(self):
        return np.ceil(float(self.totalsamples) / float(self.batch_size)).astype(np.int)

    def on_epoch_end(self):
        # Resample random indexes after each epoch
        self.indexes = np.arange(self.totalsamples)
        np.random.shuffle(self.indexes)

    def get_steps_per_epoch(self):
        return int(self.totalsamples / self.batch_size)

    # Parse the tf.train.Example proto using the feature dictionary
    def _parse_dataset_function(self, example_proto):
        return tf.io.parse_single_example(example_proto, self.feature_description)

    # load a single image from the tf.record file
    def load_record_image(self, parsed_data):
        # Get dimensions
        raw_height = tf.cast(parsed_data['image/height'], tf.int32).numpy()
        raw_width = tf.cast(parsed_data['image/width'], tf.int32).numpy()

        # Load image
        image_tf = tf.io.decode_raw(parsed_data['image/encoded'], tf.uint8).numpy()
        image_raw = cv2.imdecode(image_tf, cv2.IMREAD_COLOR)
        return image_raw.reshape((raw_height, raw_width, 3))

    def load_record_label(self, parsed_data):
        try:
            box_xmin = tf.cast(parsed_data['image/object/bbox/xmin'], tf.float32).numpy()
            box_xmax = tf.cast(parsed_data['image/object/bbox/xmax'], tf.float32).numpy()
            box_ymin = tf.cast(parsed_data['image/object/bbox/ymin'], tf.float32).numpy()
            box_ymax = tf.cast(parsed_data['image/object/bbox/ymax'], tf.float32).numpy()
            label = tf.cast(parsed_data['image/object/class/label'], tf.int64).numpy()
            ul = np.array([box_xmin, box_ymin])  # uppper left
            br = np.array([box_xmax, box_ymax])  # bottom right
        except Exception as e:
            return None, None, None
        return ul, br, label

    # resize an image while maintaining aspect ratios
    def ResizeCrop(self, img, dim_out, ul, br):
        dim_img = img.shape

        ratio_height = dim_out[0] / dim_img[0]
        ratio_width = dim_out[1] / dim_img[1]

        # determine which ratio to use
        ratio = ratio_width
        if (abs(1.0 - ratio_height) < abs(1.0 - ratio_width)):
            ratio = ratio_height

        # scaled the image
        scaled = cv2.resize(img, dsize=(0, 0), fx=ratio, fy=ratio)
        dim_scaled = scaled.shape

        # determine offsets (one of these will be 0)
        shiftx = int((dim_scaled[1] - dim_out[1]) / 2)
        shifty = int((dim_scaled[0] - dim_out[0]) / 2)
        offset = np.array([shiftx, shifty])

        output = np.zeros((dim_out[0], dim_out[1], 3), np.uint8)
        if (shiftx < 0 or shifty < 0):
            # fill black
            shiftx = abs(shiftx)
            shifty = abs(shifty)
            output[shifty:(shifty + dim_scaled[0]), shiftx:(shiftx + dim_scaled[1]), :] = scaled
        else:
            # crop
            output = scaled[shifty:(shifty + dim_out[0]), shiftx:(shiftx + dim_out[1]), :]

        # Get new bounding box percents in cropped image
        wh = np.array([dim_scaled[1], dim_scaled[0]])
        new_wh = np.array([dim_out[1], dim_out[0]])
        new_ul = ((ul * wh) - offset) / new_wh
        new_br = ((br * wh) - offset) / new_wh

        return output, new_ul, new_br

    def convert_to_yolo_output(self, ul, br, label):
        output_tensor = np.zeros(self.output_dims)
        if (ul is None or br is None or label is None):
            return output_tensor  # Empty

        center = (ul + br) / 2.0

        center_grid = center * np.array(self.output_dims[0:2])
        coords = center_grid.astype(np.int32)

        classes = np.zeros((self.num_classes))
        classes[label] = 1.0

        class_offset = self.num_classes
        box_offset = class_offset + 4

        # [classes, boxes <4>, confidences <1>]

        output_tensor[coords[1], coords[0], 0:class_offset] = classes

        output_tensor[coords[1], coords[0], class_offset:box_offset] = np.concatenate((center_grid - coords, br - ul))
        output_tensor[coords[1], coords[0], box_offset] = 1.0

        return output_tensor

    # superclass function to get next batch
    def __getitem__(self, idx):
        images = np.empty((self.batch_size, *self.image_dims))
        labels = np.empty((self.batch_size, *self.output_dims))

        # Apply feature_description dictionary
        parsed_dataset = self.raw_dataset.take(10).map(self._parse_dataset_function)

        # Sample load of 10 features
        for ii, parsed_data in enumerate(parsed_dataset):
            # load the image
            image_full = self.load_record_image(parsed_data)

            # Load bounding box and label info
            ul, br, label = self.load_record_label(parsed_data)

            # resize this to be 448, 448
            image_resized, ul, br = self.ResizeCrop(image_full, self.image_dims, ul, br)

            # store image and label
            images[ii,] = image_resized
            labels[ii,] = self.convert_to_yolo_output(ul, br, label)

        return images, labels
