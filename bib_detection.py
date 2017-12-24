from PIL import Image
import numpy as np
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

graph_path  = 'data/object_detection/bib_detection_frozen_inference_graph.pb'
labels_path = 'data/object_detection/label_map.pbtxt'

def load_detection_graph():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph

label_map = label_map_util.load_labelmap(labels_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    # Unfortunately, we can't use OpenCV to load the image directly into a
    # numpy array because the values have some differences when you load an
    # image using PIL and OpenCV. As we used PIL for training, we have to use
    # PIL for inference.
    #
    # This thread may be related: https://git.io/vbDlE
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def detect_bibs(image_path):
    """
    Returns a list of images as np arrays, one per each bib found.
    """
    bibs = []
    detection_graph = load_detection_graph()
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        pil_image = Image.open(image_path)
        w, h = pil_image.size
        image_np = load_image_into_numpy_array(pil_image)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})

        boxes, scores, classes, num = map(np.squeeze, [boxes, scores, classes, num])

        for i in range(int(num)):
            if scores[i] < 0.1: continue
            score = int(scores[i] * 1000)

            ymin, xmin, ymax, xmax = boxes[i]

            bibs.append({
                "top":      int(round(ymin * h)),
                "left":     int(round(xmin * w)),
                "right":    int(round(xmax * w)),
                "bottom":   int(round(ymax * h)),
                "score":    scores[i]
            })

    return bibs
