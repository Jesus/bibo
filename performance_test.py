import time
from glob import glob
from random import shuffle

from inference import run_inference_on_batch

image_paths = glob('data/test/**/*.jpg', recursive=True)

batch_sizes = [1,2,3,4,5,6]

while True:
    for batch_size in batch_sizes:
        time_start = time.time()

        batch_images = image_paths
        shuffle(batch_images)
        batch_images = batch_images[:batch_size]

        results = run_inference_on_batch(batch_images)

        elapsed_seconds = int(time.time() - time_start)
        seconds_per_image = elapsed_seconds / batch_size
        print("Batch size: %-5d / t: %d (%d p/i)" %
                (batch_size, elapsed_seconds, seconds_per_image))

        # for idx, image_path in enumerate(batch_images):
        #     print(image_path)
        #     for bib in results[idx]:
        #         print(f"  {bib['number']['text']}")
        #     print("")
