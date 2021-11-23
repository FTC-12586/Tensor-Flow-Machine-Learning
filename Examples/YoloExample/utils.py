import cv2
import numpy as np
import tensorflow.keras.backend as K


# Convert pbtxt file into dictionary
def read_label_map(label_map_path):
    item_id = None
    item_name = None
    items = {}

    with open(label_map_path, "r") as file:
        for line in file:
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id" in line:
                item_id = int(line.split(":", 1)[1].strip())
            elif "name" in line:
                item_name = line.split(":", 1)[1].replace("'", "").strip()

            if item_id is not None and item_name is not None:
                items[item_name] = item_id
                item_id = None
                item_name = None

    return items


def DecodeYoloOutput(output, threshold=0.6, nb_boxes=1):
    out_shape = output.shape
    grid_shape = out_shape[0:2]
    class_offset = nb_boxes * 5

    results = []  # list of labels and bounding boxes

    for yy in range(out_shape[0]):
        for xx in range(out_shape[1]):
            vec = output[yy, xx, :]
            if vec[4] > threshold:
                # Center X,Y
                xy = (vec[0:2] + np.array([xx, yy])) / out_shape[0:2]
                # Width, Height
                wh = vec[2:4]
                ul = xy - wh / 2.0
                br = xy + wh / 2.0
                class_index = np.argmax(vec[class_offset:])
                results.append({'class': class_index, 'ul': ul, 'br': br})
            elif nb_boxes > 1 and vec[9] > threshold:
                # Center X,Y
                xy = (vec[5:7] + np.array([xx, yy])) / out_shape[0:2]
                # Width, Height
                wh = vec[7:9]
                ul = xy - wh / 2.0
                br = xy + wh / 2.0
                class_label = np.argmax(vec[class_offset:])
                results.append({'class': class_index, 'ul': ul, 'br': br})

    return results


def DrawYoloOutput(image, output):
    canvas = np.copy(image).astype(np.uint8)
    boxes = DecodeYoloOutput(output)

    image_wh = np.array([canvas.shape[1], canvas.shape[0]])

    for box in boxes:
        # Convert bounding box to pixel values for ploting
        ul_px = (box['ul'] * image_wh).astype(np.int32)
        br_px = (box['br'] * image_wh).astype(np.int32)
        cv2.rectangle(canvas, ul_px, br_px, (255, 0, 0), 2)
    return canvas


# Utilities for YOLO Loss functions

class YOLO_Loss:
    def __init__(self, num_classes=20):
        self.num_classes = num_classes
        self.box_offset = num_classes + 4
        self.trust_offset = num_classes + 2

    def xywh2minmax(self, xy, wh):
        xy_min = xy - wh / 2
        xy_max = xy + wh / 2

        return xy_min, xy_max

    def iou(self, pred_mins, pred_maxes, true_mins, true_maxes):
        intersect_mins = K.maximum(pred_mins, true_mins)
        intersect_maxes = K.minimum(pred_maxes, true_maxes)
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        pred_wh = pred_maxes - pred_mins
        true_wh = true_maxes - true_mins
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
        true_areas = true_wh[..., 0] * true_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = intersect_areas / union_areas

        return iou_scores

    def yolo_head(self, feats):
        # Dynamic implementation of conv dims for fully convolutional model.
        conv_dims = K.shape(feats)[1:3]  # assuming channels last
        # In YOLO the height index is the inner most iteration.
        conv_height_index = K.arange(0, stop=conv_dims[0])
        conv_width_index = K.arange(0, stop=conv_dims[1])
        conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

        # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
        # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
        conv_width_index = K.tile(K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
        conv_width_index = K.flatten(K.transpose(conv_width_index))
        conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
        conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
        conv_index = K.cast(conv_index, K.dtype(feats))

        conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

        box_xy = (feats[..., :2] + conv_index) / conv_dims * 448
        box_wh = feats[..., 2:4] * 448

        return box_xy, box_wh

    # Compute loss between output and true labels
    def yolo_loss(self, y_true, y_pred):
        label_class = y_true[..., :self.num_classes]  # ? * 7 * 7 * 20
        label_box = y_true[..., self.num_classes:self.box_offset]  # ? * 7 * 7 * 4
        response_mask = y_true[..., self.box_offset]  # ? * 7 * 7
        response_mask = K.expand_dims(response_mask)  # ? * 7 * 7 * 1

        predict_class = y_pred[..., :self.num_classes]  # ? * 7 * 7 * 20
        predict_trust = y_pred[..., self.num_classes:self.trust_offset]  # ? * 7 * 7 * 2
        predict_box = y_pred[..., self.trust_offset:]  # ? * 7 * 7 * 8

        _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
        _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

        label_xy, label_wh = self.yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
        label_xy = K.expand_dims(label_xy, 3)  # ? * 7 * 7 * 1 * 1 * 2
        label_wh = K.expand_dims(label_wh, 3)  # ? * 7 * 7 * 1 * 1 * 2
        label_xy_min, label_xy_max = self.xywh2minmax(label_xy,
                                                      label_wh)  # ? * 7 * 7 * 1 * 1 * 2, ? * 7 * 7 * 1 * 1 * 2

        predict_xy, predict_wh = self.yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2
        predict_xy = K.expand_dims(predict_xy, 4)  # ? * 7 * 7 * 2 * 1 * 2
        predict_wh = K.expand_dims(predict_wh, 4)  # ? * 7 * 7 * 2 * 1 * 2
        predict_xy_min, predict_xy_max = self.xywh2minmax(predict_xy,
                                                          predict_wh)  # ? * 7 * 7 * 2 * 1 * 2, ? * 7 * 7 * 2 * 1 * 2

        iou_scores = self.iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? * 7 * 7 * 2 * 1
        best_ious = K.max(iou_scores, axis=4)  # ? * 7 * 7 * 2
        best_box = K.max(best_ious, axis=3, keepdims=True)  # ? * 7 * 7 * 1

        box_mask = K.cast(best_ious >= best_box, K.dtype(best_ious))  # ? * 7 * 7 * 2

        no_object_loss = 0.5 * (1 - box_mask * response_mask) * K.square(0 - predict_trust)
        object_loss = box_mask * response_mask * K.square(1 - predict_trust)
        confidence_loss = no_object_loss + object_loss
        confidence_loss = K.sum(confidence_loss)

        class_loss = response_mask * K.square(label_class - predict_class)
        class_loss = K.sum(class_loss)

        _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
        _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

        label_xy, label_wh = self.yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
        predict_xy, predict_wh = self.yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2

        box_mask = K.expand_dims(box_mask)
        response_mask = K.expand_dims(response_mask)

        box_loss = 5 * box_mask * response_mask * K.square((label_xy - predict_xy) / 448)
        box_loss += 5 * box_mask * response_mask * K.square((K.sqrt(label_wh) - K.sqrt(predict_wh)) / 448)
        box_loss = K.sum(box_loss)

        loss = confidence_loss + class_loss + box_loss

        return loss
