import os
import tempfile
import time
import uuid
import socket
import urllib.request
from shutil import rmtree

import psycopg2

from pg_interface.result import Result

class Adapter:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.worker_id = ":".join([
                str(os.getpid()),
                socket.gethostname(),
                uuid.uuid4().hex[:8]])

        self.__init_connection()
        self._reset_batch_data()

    def __init_connection(self):
        self.connection = psycopg2.connect("dbname=bibo-web-dev")

    def _reset_batch_data(self):
        self.batch_images = None
        self.local_paths = None

    def fetch_batch(self):
        # Make sure we don't hold any images already before fetching more...
        if self.batch_images != None:
            raise RuntimeError("Previously fetched batch hasn't been released")

        self._lock_batch()
        self._set_batch_images()
        self._download_batch()

        return self.local_paths

    def _lock_batch(self):
        """Acquires the lock for a batch of images"""
        cursor = self.connection.cursor()

        # Try to acquire batch_size images
        query = """
                UPDATE photos
                    SET bib_processing_at=now(), bib_processing_by='%s'
                    WHERE id IN (
                        SELECT id FROM photos
                            WHERE bib_processing_url IS NOT NULL
                              AND bib_processing_at IS NULL
                              AND bib_processed_at IS NULL
                              ORDER BY id ASC
                              LIMIT %d
                    )
                """ % (self.worker_id, self.batch_size)

        cursor.execute(query)
        self.connection.commit()
        cursor.close()

    def _set_batch_images(self):
        cursor = self.connection.cursor()

        # Fetch the IDs of the images we've been able to acquire
        query = """
                SELECT id, bib_processing_url FROM photos
                    WHERE bib_processing_by = '%s'
                """ % (self.worker_id)
        cursor.execute(query)
        self.batch_images = cursor.fetchall()
        cursor.close()

    def _download_batch(self):
        # Make sure we don't have any images already before downloading more...
        if self.local_paths != None:
            raise RuntimeError("Previously fetched batch hasn't been released")

        self.tmp_path = tempfile.mkdtemp()
        self.local_paths = []

        for i, row in enumerate(self.batch_images):
            id, url = row

            file_name = "batch-%04d.jpg" % i
            file_path = os.path.join(self.tmp_path, file_name)

            urllib.request.urlretrieve(url, file_path)
            self.local_paths.append(file_path)

    def persist_results(self, results):
        for i, inference_result in enumerate(results):
            photo_id, photo_url = self.batch_images[i]
            Result(inference_result).persist(self.connection, photo_id)

    def release_batch(self):
        """Releases the lock for our batch of images"""
        cursor = self.connection.cursor()

        query = """
                UPDATE photos
                    SET bib_processing_at = NULL, bib_processing_by = NULL
                    WHERE bib_processing_by = '%s'
                """ % (self.worker_id)

        cursor.execute(query)
        self.connection.commit()
        cursor.close()
        self._reset_batch_data()

    def _clear_tmp_dir(self):
        rmtree(self.tmp_path)

    def close(self):
        self.connection.close()


# def download_photos(urls):
#     print("")
#
# def lock_photos(batch_size):
#     """Locks `batch_size` photos and returns their IDs"""
#     cursor = connection.cursor()
#
#
#     # Fetch the IDs of the images we've been able to acquire
#     query = """
#             SELECT id, bib_processing_url FROM photos
#                 WHERE bib_processing_by = '%s'
#             """ % (worker_id)
#     cursor.execute(query)
#     connection.commit()
#
#     return cursor.fetchall()
#
# def release_photos(ids):
#     print("")
#     # TODO
#
# def get_photos(batch_size):
#     rows = lock_photos(batch_size)
#     urls = list(map(lambda x: x[0], rows))
#     paths = download_photos(urls)
#     print(rows)
