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
def bibs_data(coordinates, numbers):
    bibs = []

    for i in range(len(coordinates)):
        bibs.append({
            "coordinates": coordinates[i],
            "number": {
                "text": numbers[i]["text"],
                "scores": numbers[i]["chars_logits"]}})

    return bibs

def run_inference(image_path):
    # Bib detection
    coordinates = detect_bibs(image_path)

    # Number recognition
    numbers = detect_numbers(image_path, coordinates)

    return bibs_data(coordinates, numbers)

def run_inference_on_batch(image_paths):
    processed_images = []

    # Bib detection
    bib_detections = detect_bibs(image_paths)

    for i, image_path in enumerate(image_paths):
        # Get bib coordinates on current image
        coordinates = bib_detections[i]

        # Number recognition
        numbers = detect_numbers(image_path, coordinates)

        processed_images.append(bibs_data(coordinates, numbers))

    return processed_images
