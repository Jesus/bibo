import os
import tempfile
import time
import uuid
import socket
import urllib.request
from shutil import rmtree

import psycopg2

class Result:
    def __init__(self, data):
        self.result_data = data

    def persist(self, connection, photo_id):
        """Persists the inference results in the database"""
        cursor = connection.cursor()

        cursor.execute("""
                UPDATE photos
                    SET bib_processed_at=now()
                    WHERE id = %i
                """ % (photo_id))

        for bib in self.result_data:
            # TODO: Fill `details` column with the raw inference data
            cursor.execute("""
                INSERT INTO bibs (photo_id, x0, y0, x1, y1, number)
                    VALUES (%i, %i, %i, %i, %i, '%s')
                """ % (
                    photo_id,
                    bib['coordinates']['xmin'],
                    bib['coordinates']['ymin'],
                    bib['coordinates']['xmax'],
                    bib['coordinates']['ymax'],
                    bib['number']['text']
                ))

        connection.commit()
        cursor.close()
