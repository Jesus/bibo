import time

from pg_interface.adapter import Adapter
from inference import run_inference_on_batch

batch_size = 2

adapter = Adapter(batch_size)

while True:
    images = adapter.fetch_batch()

    if len(images) == 0:
        time.sleep(5)
    else:
        results = run_inference_on_batch(images)

        # for idx, path in enumerate(images):
        #     print("IMAGE %04d:" % idx)
        #     for bib in results[idx]:
        #         print(f"  {bib['number']['text']}")
        #     print("")

        # adapter.persist_results(results)
        adapter.release_batch()


adapter.close()
