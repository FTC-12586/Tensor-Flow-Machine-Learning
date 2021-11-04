 # Tensor-Flow-Machine-Learning
We are training a .tflite model for use on the rev hub, we are importing a provided data set from the ftc-ml online tool.

Information:
  * The architecture of the outputted .tflite model is: __TBD__
  
  * The data set layout is: 
    * "The datasets are downloaded as one or more zip files. The zip files contain TensorFlow record files."(https://community.ftclive.org/t/downloading-datasets/34)
  
# To do:
- [ ] Figure out how to open the .tflite file given to us

- [ ] Successfully import data set:
 * Load Batches of TF Images and bboxes
 * Make a 1 frame data set for analysis of packing method
 * Find out why it works with python 3.8 instead of 3.9
- [ ] Determine Architecture:
 * Look at SSD MobileNet v2 320x320 model
 * Look at SSD Models
 * Look at YOLO Models
 * Currently, on-device inference is only optimized with SSD models
 * Think about using pretrained model for cross training benifits
- [ ] Train from architecture
 * The tensorflow library has a tf.train method
 * tf compares bboxes
 * tf handels back propagation
 * we do not have to match architecture with the one used in the robot

#To install Tensorflow:
run these commands in the pycharm terminal window:
  * conda create -n tf tensorflow
  * conda activate tf
  * pip install opencv-python 
  * pip install Pillow

Then switch your interpreter to the conda/Python 3.9 (tf) option

Maybe not this command: pip install --ignore-installed --upgrade tensorflow==2.5.0

#Resources:
* https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset
* https://www.tensorflow.org/lite/tutorials/model_maker_object_detection
* https://www.tensorflow.org/lite/examples/object_detection/overview#model_customization
* https://www.tensorflow.org/lite/examples/object_detection/overview
* https://github.com/kerrj/yoloparser/blob/master/README.md#how-can-i-train-yolo-on-custom-objects
* https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#mobile-models
* https://github.com/FIRST-Tech-Challenge/fmltc/tree/8236d6fbc72ae5e2dde0ed018cd8560ebd5ee360


#Information:
* The ftc-ml tool uses the SSD MobileNet v2 320x320 model as its default starter model from the tensorflow zoo
* Even though TensorFlow isn’t the best at recognizing geometries, it’s incredibly good at recognizing
textures. No, probably not the kinds of textures you’re thinking about – we’re talking visual textures like
zebra stripes, giraffe spots, neon colors, and so on. Colored patterns are TensorFlow’s strength.
Careful Team Shipping Element design beforehand may yield great benefits later.
* try your best to video how your robot will see the objects in competition, and try your best in competition to make 
sure that your robot only sees the objects like you trained the model
* Generally after training if your model is predicting all objects above 50% all the time you’re doing really well
* Repo for FTC-Ml Tool: https://github.com/FIRST-Tech-Challenge/fmltc
* Currently, on-device inference is only optimized with SSD models