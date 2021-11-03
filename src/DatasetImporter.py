import io
import os
import cv2
from zipfile import ZipFile
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
    def load_dataset(filename):
        print("loading record names: ", filename)

        filenames = [filename]
        raw_dataset = tf.data.TFRecordDataset(filenames)

        # Sample load of 10 features
        for raw_record in raw_dataset.take(10):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())

            sourceFile = open('features.txt', 'w')
            print(example, file=sourceFile)
            sourceFile.close()
            # print(example)

            parsed_data = tf.io.parse_single_example(raw_record, DatasetImporter.feature_description)

            raw_height = tf.cast(parsed_data['image/height'], tf.int32)
            raw_width = tf.cast(parsed_data['image/width'], tf.int32)
            image_tf = tf.io.decode_raw(parsed_data['image/encoded'], tf.uint8)

            print(raw_height)
            print(raw_width)

            image_tf = tf.reshape(image_tf, [raw_height, raw_width, 3])
            image_tf = tf.image.resize_images(image_tf, size=[self.height, self.width])

            with tf.Session() as sess:
                image = sess.run(image_tf)

            cv2.imshow("tf", image)
            cv2.waitKey(0)

    @staticmethod
    def _parse_record_file(file_contents: io.BytesIO) -> list:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        raw_dataset = tf.data.TFRecordDataset(file_contents.read())
        parsed_data = tf.io.parse_single_example(raw_dataset, DatasetImporter.feature_description)
        breakpoint()

    @staticmethod
    def _parse_pbtxt_file(file_contents: io.BytesIO) -> dict:
        # reads in the data from the file object as a string
        data = file_contents.read().decode()

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
        path, ext = os.path.splitext(file_name)
        ext = str(ext)
        ext = ext.strip()
        ext = ext.strip("'")
        if ext == '.pbtxt' or ext == "'.pbtxt'" or ext == ".pbtxt":
            return True
        return False

    @staticmethod
    def load(file_name: str):
        z = ZipFile(file_name, 'r')
        record_files = []
        for file in z.filelist:
            if DatasetImporter._is_pbtxt(file.filename):
                label = DatasetImporter._parse_pbtxt_file(z.open(file))
            elif DatasetImporter._is_record_file(file.filename):

                record_files.append(DatasetImporter._parse_record_file(z.open(file)))
            else:
                # sanity check on the file
                breakpoint()
                raise ValueError("The File is Malformed")
        z.close()
