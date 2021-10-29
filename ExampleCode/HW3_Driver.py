import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import cv2

from CatalogDataset import CatalogDataset
from CustomNetwork import CustomNetwork
from ImageGenerator import ImageGenerator
from ImagePreprocess import ImagePreprocess

# Assignment #3
# Solution Manual
# Rod Dockter - ME 5286

# input dataset
image_folder = './Dataset'
input_shape = (240, 320, 3)

# Problem 1 - Image Preprocessing
preprocess = ImagePreprocess(input_shape, 'BGR')
image_file = './Dataset/black_bear/0a8eaec98cc2542440.jpg'
image = preprocess.preprocess(image_file)
augmented = preprocess.augment(image.copy())

cv2.imshow("Image", image)
cv2.imshow("Augmented", augmented)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)

# Problem 2 - Load dataset
catalog = CatalogDataset(image_folder, 0.8)
data, labels = catalog.get_dataset()
num_classes = catalog.NumClasses()

print("\n\n Data Catalog:")
print(data)
print(labels)
print(catalog.ListLabels())

# Problem 3 - Data Generators
training_generator = ImageGenerator(data['train'], labels, image_folder, num_classes, batch_size=32, dim=input_shape,
                                    augment=True, shuffle=True)
validation_generator = ImageGenerator(data['test'], labels, image_folder, num_classes, batch_size=32, dim=input_shape,
                                      augment=False, shuffle=False)

print("\n\n Batch:")
x, y = validation_generator.__getitem__(0)
print(x)
print(y)
print(x.shape)
print(y.shape)

# Problem 4 - Build Network
stored_model = 'trained_model.h5'

network = CustomNetwork(num_classes=num_classes, input_shape=input_shape)
print("\n\n Model Summary:")
network.model.summary()

print("\n\n Test Forward Pass:")
result = network.model(image[None,]).numpy()
print(result)

# Problem 5 - Training Network
print("\n\n Training Model (~1 hour):")
network.TrainModel(training_generator, validation_generator, epochs=50, preload=True, stored_model=stored_model)

# Problem 6 - Inference a single file
file = './Dataset/red_fox/0f16b39124e3a693ba.jpg'
network.LoadModel(stored_model=stored_model)
index_estimate = network.InferenceModel(file)
label_estimate = catalog.classes[index_estimate]

print("\n\n Inference:")
print(file)
print(label_estimate)
