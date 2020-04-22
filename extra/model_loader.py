import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class GrappleDetection(object):
    def __init__(self, path_to_model):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            #od_graph_def = tf.compat.v1.GraphDef()
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_model, "rb") as fid:
            #with tf.compat.v2.io.gfile.GFile(path_to_model, "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")
            self.image_tensor = self.detection_graph.get_tensor_by_name(
                "image_tensor:0"
            )
            self.d_boxes = self.detection_graph.get_tensor_by_name("detection_boxes:0")
            self.d_scores = self.detection_graph.get_tensor_by_name(
                "detection_scores:0"
            )
            self.d_classes = self.detection_graph.get_tensor_by_name(
                "detection_classes:0"
            )
            self.num_d = self.detection_graph.get_tensor_by_name("num_detections:0")
        self.sess = tf.Session(graph=self.detection_graph)

    def predict(self, img):
        with self.detection_graph.as_default():
            img_expanded = np.expand_dims(img, axis=0)
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded},
            )
        return boxes, scores, classes, num

    def _transform_to_image_size(self, boxe, image_width=648, image_height=1024):
        ymin_detection = int((boxe[0] * image_height))
        xmin_detection = int((boxe[1] * image_width))
        ymax_detection = int((boxe[2] * image_height))
        xmax_detection = int((boxe[3] * image_width))

        return np.array(
            [xmin_detection, ymin_detection, xmax_detection, ymax_detection]
        )


class LogsClassification(object):
    def __init__(self, path_to_model_meta, path_to_model_data):
        self.path_to_meta = path_to_model_meta
        self.path_to_model = path_to_model_data

    def predict(self, img):
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(self.path_to_meta)
            saver.restore(sess, tf.train.latest_checkpoint(self.path_to_model))
            return sess.run("Sigmoid:0", feed_dict={"X:0": img})