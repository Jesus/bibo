import os, sys
import cv2
import numpy as np
import tensorflow as tf

# Model config
graph_path = "data/attention_ocr/number_recognition_frozen_inference_graph.pb"
batch_size = 32
width, height = 80, 80

def load_bibs(image, bibs):
    images_data = np.ndarray(shape=(batch_size, width, height, 3),
            dtype='float32')

    for i, bib_coordinates in enumerate(bibs):
        ymin = bib_coordinates["ymin"]
        ymax = bib_coordinates["ymax"]
        xmin = bib_coordinates["xmin"]
        xmax = bib_coordinates["xmax"]

        if xmin == xmax or ymin == ymax:
            continue

        bib = image[ymin:ymax, xmin:xmax]
        bib = cv2.resize(bib, (width, height)) / 255.0

        images_data[i, ...] = bib

    return images_data

def load_graph():
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(graph_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    return graph

def chars_as_number(chars):
    # We use a direct mapping as charset: 0 -> "0", 1 -> "1", 2 -> "2", etc.
    # Null character is identified by 10.
    number = ""
    for char in chars:
        if char == 10:
            break
        number = "%s%i" % (number, char)

    return number

def detect_numbers(image_path, bibs):
    graph = load_graph()
    image = cv2.imread(image_path)
    images_data = load_bibs(image, bibs)

    images_placeholder = graph.get_tensor_by_name('images_placeholder:0')
    chars_logits_tensor = graph.get_tensor_by_name('output/chars_logits:0')
    chars_tensor = graph.get_tensor_by_name('AttentionOcr_v1/predicted_chars:0')

    with tf.Session(graph=graph) as sess:
        chars_logits, chars = sess.run(
                [chars_logits_tensor, chars_tensor],
                feed_dict={images_placeholder: images_data})

    numbers = []
    for i in range(len(bibs)):
        numbers.append({
                "chars_logits": chars_logits[i],
                "chars": chars[i],
                "text": chars_as_number(chars[i])})

    return numbers
