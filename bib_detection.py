from PIL import Image
import numpy as np
import tensorflow as tf

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

def load_image_into_numpy_array(image):
    # Unfortunately, we can't use OpenCV to load the image directly into a
    # numpy array because the values have some differences when you load an
    # image using PIL and OpenCV. As we used PIL for training, we have to use
    # PIL for inference.
    #
    # This thread may be related: https://git.io/vbDlE
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def load_images(image_paths):
    batch_size = len(image_paths)
    dimensions = []
    width, height = 1600, 1067
    images_data = np.ndarray(shape=(batch_size, height, width, 3),
            dtype='float32')

    for i, image_path in enumerate(image_paths):
        pil_image = Image.open(image_path)
        width, height = pil_image.size

        # TODO: Resize to average dimensions

        images_data[i, ...] = load_image_into_numpy_array(pil_image)
        dimensions.append([width, height])

    return images_data, dimensions

def detect_bibs(image_paths):
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

        images_data, dimensions = load_images(image_paths)
        boxes, scores, classes, num = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: images_data})

        for i in range(len(images_data)):
            image_bibs = []
            for j in range(int(num[i])):
                if scores[i][j] < 0.1: continue
                score = int(scores[i][j] * 1000)

                ymin, xmin, ymax, xmax = boxes[i][j]

                image_bibs.append({
                    "ymin": int(round(ymin * dimensions[i][1])),
                    "xmin": int(round(xmin * dimensions[i][0])),
                    "xmax": int(round(xmax * dimensions[i][0])),
                    "ymax": int(round(ymax * dimensions[i][1])),
                    "score": scores[i][j]
                })
            bibs.append(image_bibs)

    return bibs
