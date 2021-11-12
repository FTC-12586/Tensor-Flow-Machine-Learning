import tensorflow.keras.backend as K
import tensorflow as tf

class Yolo_Reshape(tf.keras.layers.Layer):
    def __init__(self, target_shape, num_classes, nb_boxes):
        super(Yolo_Reshape, self).__init__()
        self.target_shape = tuple(target_shape)
        self.num_classes = num_classes
        self.nb_boxes = nb_boxes

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'target_shape': self.target_shape,
            'num_classes': self.num_classes,
            'nb_boxes': self.nb_boxes
        })
        return config

    def call(self, input):
        # grids 7x7
        S = [self.target_shape[0], self.target_shape[1]]
        # classes
        C = self.num_classes
        # no of bounding boxes per grid
        B = self.nb_boxes

        idx1 = S[0] * S[1] * C
        idx2 = idx1 + S[0] * S[1] * B

        # class probabilities
        class_probs = K.reshape(input[:, :idx1], (K.shape(input)[0],) + tuple([S[0], S[1], C]))
        class_probs = K.softmax(class_probs)

        #confidence
        confs = K.reshape(input[:, idx1:idx2], (K.shape(input)[0],) + tuple([S[0], S[1], B]))
        confs = K.sigmoid(confs)

        # boxes
        boxes = K.reshape(input[:, idx2:], (K.shape(input)[0],) + tuple([S[0], S[1], B * 4]))
        boxes = K.sigmoid(boxes)

        outputs = K.concatenate([class_probs, confs, boxes])
        return outputs
