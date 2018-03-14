import urllib.request

url_prefix = "https://s3.amazonaws.com/bibo.inference/20171224/"
files = {
    "number_recognition_frozen_inference_graph.pb":
        "data/attention_ocr/number_recognition_frozen_inference_graph.pb",
    "bib_detection_frozen_inference_graph.pb":
        "data/object_detection/bib_detection_frozen_inference_graph.pb"
}

for url, path in files.items():
    url = "%s%s" % (url_prefix, url)
    print(url)
    urllib.request.urlretrieve(url, path)
