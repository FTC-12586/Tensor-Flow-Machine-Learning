import os

import cv2
import numpy as np
import tensorflow as tf


class DatasetImporter:
    feature_description = {
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

    @staticmethod
    def _parse_dataset_function(example_proto) -> dict:
        # Parse the input tf.train.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, DatasetImporter.feature_description)

    @staticmethod
    def get_dataset_size(recordfile) -> int:
        return sum(1 for _ in tf.data.TFRecordDataset(recordfile))

    @staticmethod
    def load_record_image(parsed_data) -> np.ndarray:
        # Get dimensions
        raw_height = tf.cast(parsed_data['image/height'], tf.int32).numpy()
        raw_width = tf.cast(parsed_data['image/width'], tf.int32).numpy()

        # Load image
        image_tf = tf.io.decode_raw(parsed_data['image/encoded'], tf.uint8).numpy()
        image_raw = cv2.imdecode(image_tf, cv2.IMREAD_COLOR)
        return image_raw.reshape((raw_height, raw_width, 3))

    @staticmethod
    def load_record_label(parsed_data) -> tuple:
        box_xmin = tf.cast(parsed_data['image/object/bbox/xmin'], tf.float32).numpy()
        box_xmax = tf.cast(parsed_data['image/object/bbox/xmax'], tf.float32).numpy()
        box_ymin = tf.cast(parsed_data['image/object/bbox/ymin'], tf.float32).numpy()
        box_ymax = tf.cast(parsed_data['image/object/bbox/ymax'], tf.float32).numpy()
        label = tf.cast(parsed_data['image/object/class/label'], tf.int64).numpy()
        ul = np.array([box_xmin, box_ymin])  # uppper left
        br = np.array([box_xmax, box_ymax])  # bottom right
        return ul, br, label

    @staticmethod
    def convert_to_yolo_output(ul, br, label)-> np.ndarray:
        output_tensor = np.zeros((7, 7, 25))
        center = (ul + br) / 2.0
        print(center)
        center_grid = center * np.array([7, 7])
        print(center_grid)
        coords = center_grid.astype(np.int32)
        print(coords)

        classes = np.zeros((20))
        classes[label] = 1.0

        output_tensor[coords[1], coords[0], 0:2] = center_grid - coords

        output_tensor[coords[1], coords[0], 2:4] = br - ul
        output_tensor[coords[1], coords[0], 4] = 1.0
        output_tensor[coords[1], coords[0], 5:25] = classes

        return output_tensor

    @staticmethod
    def _parse_record_file(filename: str) -> dict:

        raw_dataset = tf.data.TFRecordDataset([filename])

        parsed_dataset = raw_dataset.map(DatasetImporter._parse_dataset_function)

        listOfDatasetDictionaries = []
        # Sample load of 10 features
        for parsed_data in parsed_dataset:
            # Load bounding box and label info
            ul, br, label = DatasetImporter.load_record_label(parsed_data)

            output_tensor = DatasetImporter.convert_to_yolo_output(ul, br, label)

            image = DatasetImporter.load_record_image(parsed_data)

            image = DatasetImporter.ResizeFill(image)
            data = {
                "upper_left_bbx": ul,
                "bottom_right_bbx": br,
                "output_tensor": output_tensor,
                "image": image
            }
            listOfDatasetDictionaries.append(data)
        return listOfDatasetDictionaries

    @staticmethod
    def cv_size(img) -> tuple:
        return tuple(img.shape[1::-1])

    @staticmethod
    def ResizeFill(Img: np.ndarray) -> np.ndarray:

        width, height = DatasetImporter.cv_size(Img)
        m = 480 / width
        width = int(width * m)
        height = int(height * m)
        if height % 2 != 0:
            height = height + 1
        resized = cv2.resize(Img, (width, height))

        width, height = DatasetImporter.cv_size(resized)
        assert width == 480
        assert height <= 480

        bordersize = int((480 - height) / 2)

        resized = cv2.copyMakeBorder(
            resized,
            top=bordersize,
            bottom=bordersize,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT
        )

        width, height = DatasetImporter.cv_size(resized)
        assert width == height == 480
        return resized

    @staticmethod
    def ResizeCrop(img, dim_out) -> np.ndarray:
        dim_img = img.shape

        ratio_height = dim_out[0] / dim_img[0]
        ratio_width = dim_out[1] / dim_img[1]

        # determine which ratio to use
        ratio = ratio_width
        if abs(1.0 - ratio_height) < abs(1.0 - ratio_width):
            ratio = ratio_height

        # scaled the image
        scaled = cv2.resize(img, dsize=(0, 0), fx=ratio, fy=ratio)
        dim_scaled = scaled.shape

        # determine offsets (one of these will be 0)
        shiftx = int((dim_scaled[1] - dim_out[1]) / 2)
        shifty = int((dim_scaled[0] - dim_out[0]) / 2)

        output = np.zeros((dim_out[0], dim_out[1], 3), np.uint8)
        if shiftx < 0 or shifty < 0:
            # fill black
            shiftx = abs(shiftx)
            shifty = abs(shifty)
            output[shifty:(shifty + dim_scaled[0]), shiftx:(shiftx + dim_scaled[1]), :] = scaled
        else:
            # crop
            output = scaled[shifty:(shifty + dim_out[0]), shiftx:(shiftx + dim_out[1]), :]

        return output

    @staticmethod
    def _parse_pbtxt_file(filename: str) -> dict:
        # reads in the data from the file object as a string
        with open(filename, 'r') as f:
            data = f.read()

        # splits the data into sections
        data = data.split('\n')

        # removes all lines except the id and name lines
        for i in range(0, len(data)):
            data[i] = data[i].strip()
        while 'item {' in data:
            data.remove('item {')
        while '}' in data:
            data.remove('}')
        while '' in data:
            data.remove('')

        ids = []
        names = []

        # fills out a list of names and id's so that id[x] is the id of name[x]
        for i in range(0, len(data)):
            if 'id: ' in data[i]:
                ids.append(data[i])
            elif 'name:' in data[i]:
                names.append(data[i])
            else:
                raise ValueError(
                    "The File is Malformed")  # sanity check on the file as it should never reach this point

        # sanity check on the file
        if len(ids) != len(names):
            raise ValueError("The File is Malformed")

        d = {}
        for i in range(0, len(ids)):
            d[ids[i].split(': ')[1]] = names[i].split("'")[1]  # makes the id the dictionary key and the name the value
        return d

    @staticmethod
    def _is_record_file(file_name: str) -> bool:
        path, ext = os.path.splitext(file_name)
        ext = ext.split('-')[0].strip()
        if ext == '.record':
            return True
        return False

    @staticmethod
    def _is_pbtxt(file_name: str) -> bool:
        try:
            path, ext = os.path.splitext(file_name)
            ext = str(ext)
            ext = ext.strip()
            ext = ext.strip("'")
            if ext == '.pbtxt' or ext == "'.pbtxt'" or ext == ".pbtxt":
                return True
            return False
        except ValueError:
            return False

    @staticmethod
    def absoluteFilePaths(directory) -> list:
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                tmp = os.path.abspath(os.path.join(dirpath, f))
                yield tmp

    @staticmethod
    def loadFile(fileName: str) -> dict:
        fileName = os.path.abspath(fileName)
        if DatasetImporter._is_pbtxt(fileName):
            try:
                return DatasetImporter._parse_pbtxt_file(fileName)
            except Exception as e:
                pass

        return DatasetImporter._parse_dataset_function(fileName)

    @staticmethod
    def loadFolder(folder_name: str) -> list:
        folder_name = os.path.abspath(folder_name)

        folder_files = [file for file in DatasetImporter.absoluteFilePaths(folder_name)]
        record_datasets = []
        print(folder_files)
        for file in folder_files:
            if DatasetImporter._is_pbtxt(file):
                label = DatasetImporter._parse_pbtxt_file(file)
                print("parsed .pbtxt")
                continue

            elif DatasetImporter._is_record_file(file):
                record_datasets.append(DatasetImporter._parse_record_file(file))
                continue
            else:
                # sanity check on the file
                raise ValueError("The File is Malformed")
        return [record_datasets, label]
