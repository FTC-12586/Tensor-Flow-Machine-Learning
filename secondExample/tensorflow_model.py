import tensorflow as tf

# Load TFLite model and allocate tensors.

path = "/home/rodney/Downloads/FreightFrenzy_BC.tflite"
interpreter = tf.lite.Interpreter(model_path=path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# get details for each layer
all_layers_details = interpreter.get_tensor_details()

print("Input Shape: {}".format(input_details[0]['shape']))
print("Output Shape: {}".format(output_details[0]['shape']))


# Based on this I believe this model is an SSD Mobile Net V2 Architecture
