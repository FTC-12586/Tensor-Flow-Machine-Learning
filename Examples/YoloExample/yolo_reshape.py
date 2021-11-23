import tensorflow as tf
import tensorflow.keras.backend as k_backend


class Yolo_Reshape(tf.keras.layers.Layer):
    def __init__(self, target_shape, num_classes, nb_boxes):
        super(Yolo_Reshape, self).__init__()
        self.target_shape = tuple(target_shape)
        self.num_classes = num_classes
        self.nb_boxes = nb_boxes

    def get_config(self):
        config: dict = super().get_config().copy()
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
        number_of_classes = self.num_classes
        # no of bounding boxes per grid
        number_of_bboxes = self.nb_boxes

        idx1 = S[0] * S[1] * number_of_classes
        idx2 = idx1 + S[0] * S[1] * number_of_bboxes

        # class probabilities
        class_probs = k_backend.reshape(input[:, :idx1], (k_backend.shape(input)[0],) + tuple([S[0], S[1], number_of_classes]))
        class_probs = k_backend.softmax(class_probs)

        # confidence
        confs = k_backend.reshape(input[:, idx1:idx2], (k_backend.shape(input)[0],) + tuple([S[0], S[1], number_of_bboxes]))
        confs = k_backend.sigmoid(confs)

        # boxes
        boxes = k_backend.reshape(input[:, idx2:], (k_backend.shape(input)[0],) + tuple([S[0], S[1], number_of_bboxes * 4]))
        boxes = k_backend.sigmoid(boxes)

        outputs = k_backend.concatenate([class_probs, confs, boxes])
        return outputs
