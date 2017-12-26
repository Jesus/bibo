from bib_detection import detect_bibs
from number_recognition import detect_numbers

# Runs inference on the given image. The result is an array of bibs, each one
# containing "coordinates" & "number". Here's how the data is organized:
# [{
#     "coordinates": {
#         "xmin": ...,
#         "ymin": ...,
#         "xmax": ...,
#         "ymax": ...,
#         "score": ...
#     },
#     "number": {
#         "text": ...,
#         "scores": []
#     }
# }, ...]
def run_inference(image_path):
    bibs = []

    # Bib detection
    coordinates = detect_bibs(image_path)

    # Number recognition
    numbers = detect_numbers(image_path, coordinates)

    for i in range(len(coordinates)):
        bibs.append({
            "coordinates": coordinates[i],
            "number": {
                "text": numbers[i]["text"],
                "scores": numbers[i]["chars_logits"]}})

    return bibs

def run_inference_on_batch(image_paths):
    processed_images = []

    # Bib detection
    bib_detections = detect_bibs(image_paths)

    for i, image_path in enumerate(image_paths):
        # Get bib coordinates on current image
        coordinates = bib_detections[i]

        # Number recognition
        numbers = detect_numbers(image_path, coordinates)

        bibs = []
        for i in range(len(coordinates)):
            bibs.append({
                "coordinates": coordinates[i],
                "number": {
                    "text": numbers[i]["text"],
                    "scores": numbers[i]["chars_logits"]}})

        processed_images.append(bibs)

    return processed_images
