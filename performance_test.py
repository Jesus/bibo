from inference import run_inference_on_batch

images = [
    "data/test/mm-alicante-17/DSC_4485.jpg",
    "data/test/mm-alicante-17/DSC_4787.jpg",
    "data/test/mm-alicante-17/DSC_4433.jpg",
    "data/test/mm-alicante-17/DSC_4781.jpg"
]
images = images[:1]

results = run_inference_on_batch(images)
for idx, image_path in enumerate(images):
    print(image_path)
    for bib in results[idx]:
        print(f"  {bib['number']['text']}")
    print("")
