 # Tensor-Flow-Machine-Learning
We are training a .tflite model for use on the rev hub, we are importing a provided data set from the ftc-ml online tool.

Information:
  * The architecture of the outputted .tflite model is: __TBD__
  
  * The data set layout is: 
    * "The datasets are downloaded as one or more zip files. The zip files contain TensorFlow record files."(https://community.ftclive.org/t/downloading-datasets/34)
  
To do:
  * Successfully import data set
  * Determine Architecture
  * Train from architecture

__To install Tensorflow, run these commands in the pycharm terminal window:__
  * conda create -n tf tensorflow
  * conda activate tf
  * pip install opencv-python 
  * pip install Pillow

Then switch your interpreter to the conda/Python 3.9 (tf) option

Maybe not this command: pip install --ignore-installed --upgrade tensorflow==2.5.0
