import os

import numpy as np


# Assignment #3, Problem 2
# Solution Manual
# Rod Dockter - ME 5286


# Requires specific folder layout:
# Data/
#   class1/
#       sample1.png
#       sample2.png
#   class2/
#       sample1.png
#       sample2.png

class CatalogDataset():
    # Load dataset training and testing portions
    def __init__(self, path, split=0.8, shuffle=True):
        # Variables
        self.split = split  # float
        self.shuffle = shuffle  # bool
        self.path = path  # string

        # dictionaries
        self.labels = {}
        self.samples = {}
        self.classes = []
        all_files = []
        self.nb_classes = 0

    def get_dataset(self):
        # Get all classes
        for class_dir in os.listdir(self.path):
            # Full Path
            dir = os.path.join(self.path, class_dir)
            if not os.path.isdir(dir):
                continue
            self.classes.append(class_dir)

        # Total number of classes
        self.nb_classes = len(self.classes)

        self.all_files = []
        # Get all files
        for class_dir in self.classes:
            # Find files in this class
            dir = os.path.join(self.path, class_dir)
            for file in os.listdir(dir):
                if os.path.isfile(os.path.join(dir, file)):
                    # If it's a file, add it to list
                    full_file = os.path.join(class_dir, file)
                    self.all_files.append(full_file)
                    self.labels[full_file] = self.classes.index(class_dir)

        # Split
        self.samples['train'], self.samples['test'] = self.__TrainTestSplit(self.all_files)

        # Returns dictionary of 'train'/'test' samples and labels
        return self.samples, self.labels

    def ListLabels(self):
        return self.classes

    def NumClasses(self):
        return len(self.classes)

    def GetLabel(self, index):
        return self.classes[index]

    def GetLabelOnehot(self, onehot):
        index = np.argmax(onehot, axis=1)
        return self.classes[index[0]]

    def __TrainTestSplit(self, sample_list):
        # Get random shuffle of all sample indices
        n_samples = len(sample_list)
        indices = np.arange(n_samples, dtype=np.int32)
        if self.shuffle:
            np.random.seed(0)  # Make Repeatable
            np.random.shuffle(indices)

        # first 80% are train, remainder are test
        n_train = int(n_samples * self.split)
        train_ind = indices[0:n_train]
        test_ind = indices[n_train:]

        return [sample_list[i] for i in train_ind], [sample_list[i] for i in test_ind]
