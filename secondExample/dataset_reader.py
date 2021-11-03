import cv2
import numpy as np
import tensorflow as tf
import base64
import io

#Each element in the .record dataset has these features
feature_description={
    'image/source_id': tf.io.FixedLenFeature([], tf.string),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/height' : tf.io.FixedLenFeature([], tf.int64),
    'image/width'  : tf.io.FixedLenFeature([], tf.int64),
    'image/object/bbox/xmin'  : tf.io.FixedLenFeature([], tf.float32),
    'image/object/bbox/xmax'  : tf.io.FixedLenFeature([], tf.float32),
    'image/object/bbox/ymin'  : tf.io.FixedLenFeature([], tf.float32),
    'image/object/bbox/ymax'  : tf.io.FixedLenFeature([], tf.float32),
    'image/object/class/label'  : tf.io.FixedLenFeature([], tf.int64),
    'image/object/class/text'  : tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
}

def _parse_dataset_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)

def get_dataset_size(recordfile):
    return sum(1 for _ in tf.data.TFRecordDataset(recordfile))

def load_record_image(parsed_data):
     #Get dimensions
     raw_height = tf.cast(parsed_data['image/height'], tf.int32).numpy()
     raw_width = tf.cast(parsed_data['image/width'], tf.int32).numpy()

     #Load image
     image_tf = tf.io.decode_raw(parsed_data['image/encoded'], tf.uint8).numpy()
     image_raw = cv2.imdecode(image_tf, cv2.IMREAD_COLOR)
     return image_raw.reshape((raw_height, raw_width, 3))

def load_record_label(parsed_data):
    box_xmin = tf.cast(parsed_data['image/object/bbox/xmin'], tf.float32).numpy()
    box_xmax = tf.cast(parsed_data['image/object/bbox/xmax'], tf.float32).numpy()
    box_ymin = tf.cast(parsed_data['image/object/bbox/ymin'], tf.float32).numpy()
    box_ymax = tf.cast(parsed_data['image/object/bbox/ymax'], tf.float32).numpy()
    label = tf.cast(parsed_data['image/object/class/label'], tf.int64).numpy()
    ul = np.array([box_xmin, box_ymin]) #uppper left
    br = np.array([box_xmax, box_ymax]) #bottom right
    return ul, br, label

def convert_to_yolo_output(ul, br, label):
    output_tensor = np.zeros((7, 7, 25))
    center = (ul+br)/2.0
    print(center)
    center_grid = center * np.array([7,7])
    print(center_grid)
    coords = center_grid.astype(np.int32)
    print(coords)

    classes = np.zeros((20))
    classes[label] = 1.0

    output_tensor[coords[1], coords[0], 0:2] = center_grid - coords

    output_tensor[coords[1], coords[0], 2:4] = br-ul
    output_tensor[coords[1], coords[0], 4] = 1.0
    output_tensor[coords[1], coords[0], 5:25] =classes

    return output_tensor

#resize an image while maintaining aspect ratios
def ResizeCrop(img, dim_out):
    dim_img = img.shape

    ratio_height = dim_out[0]/dim_img[0]
    ratio_width = dim_out[1]/dim_img[1]

    #determine which ratio to use
    ratio = ratio_width
    if(abs(1.0-ratio_height) < abs(1.0-ratio_width)):
        ratio = ratio_height

    #scaled the image
    scaled = cv2.resize(img, dsize=(0,0), fx=ratio, fy=ratio)
    dim_scaled = scaled.shape

    #determine offsets (one of these will be 0)
    shiftx = int( (dim_scaled[1] - dim_out[1])/2 )
    shifty = int( (dim_scaled[0] - dim_out[0])/2 )

    output = np.zeros((dim_out[0], dim_out[1], 3), np.uint8)
    if(shiftx < 0 or shifty < 0):
        #fill black
        shiftx = abs(shiftx)
        shifty = abs(shifty)
        output[shifty:(shifty+dim_scaled[0]), shiftx:(shiftx+dim_scaled[1]), :] = scaled
    else:
        #crop
        output = scaled[shifty:(shifty+dim_out[0]), shiftx:(shiftx+dim_out[1]), :]

    return output


def load_dataset(filename):
  print("loading record names: ", filename)

  print("dataset size: {}".format(get_dataset_size(filename)))

  raw_dataset = tf.data.TFRecordDataset( [filename] )

  parsed_dataset = raw_dataset.take(10).map(_parse_dataset_function)

  #Sample load of 10 features
  for parsed_data in parsed_dataset:

      #Load bounding box and label info
      ul, br, label = load_record_label(parsed_data)

      output_tensor = convert_to_yolo_output(ul, br, label)

      image = load_record_image(parsed_data)

      #Convert bounding box to pixel values for ploting
      ul_px = (ul*np.array([image.shape[1], image.shape[0]])).astype(np.int32)
      br_px = (br*np.array([image.shape[1], image.shape[0]])).astype(np.int32)
      cv2.rectangle(image, ul_px, br_px, (255,0,0), 2)

      image = ResizeCrop(image, [448,448,3])

      cv2.imshow("tf", image)
      cv2.waitKey(0)


#Set to the record file location
load_dataset("/home/rodney/python_ws/tensorflow_frc/dataset/eval_dataset.record-00000-00001")
