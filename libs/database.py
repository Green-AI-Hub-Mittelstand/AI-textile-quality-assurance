import io
import json
import sqlite3
from typing import Any

import cv2.typing
import pandas as pd
from PIL import Image


def _rows_to_dataframe(rows):
    data_rows = []
    for row in rows:
        data = {
            "id": row["id"],
            "check_time": row["check_time"],
            "result": bool(row["result"]),
        }

        results = json.loads(row["json_data"])
        for result in results:
            data[result['check']] = result['actual']
        data_rows.append(data)

    return pd.DataFrame(data=data_rows)


class QualityCheckDB:
    def __init__(self, db_name='quality_check.db'):
        """ Initialize the database connection and create table if not exists. """
        self.db_name = db_name
        self.create_table()

    def create_table(self):
        """ Create the quality_checks table if it doesn't exist. """
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                check_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL, 
                result INTEGER NOT NULL,
                json_data TEXT NOT NULL,
                image BLOB
            )''')
        conn.commit()
        conn.close()


    def insert_quality_check(self, result: bool, rows: list[dict[str, Any]], image_cropped: cv2.typing.MatLike,
                             image_ext = '.jpg', image_resize=True):
        """ Insert a new quality check result along with the image into the database. """

        if image_resize:
            image_cropped = cv2.resize(image_cropped, None, fx=0.25, fy=0.25)

        _, buffer = cv2.imencode(image_ext, image_cropped)
        json_string = json.dumps(rows)

        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        # Insert the JSON data and image into the table
        cursor.execute('INSERT INTO quality_checks (result, json_data, image) VALUES (?, ?, ?)',
                       (result, json_string, buffer.tobytes()))
        conn.commit()
        conn.close()

    def retrieve_quality_checks(self, time_limit="-7 days"):
        """Retrieve and return all quality check data (except images) from the database."""
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Fetch all rows from the quality_checks table
        cursor.execute(f"SELECT id, check_time, result, json_data "
                       f"FROM quality_checks "
                       f"WHERE check_time > datetime('now', '{time_limit}');")

        df = _rows_to_dataframe(cursor.fetchall())
        conn.close()

        try:
            df["check_time"] = pd.to_datetime(df["check_time"])
        except KeyError:
            raise RuntimeError("No data found!")

        return df.set_index("check_time")

    def retrieve_image(self, id_: int):
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT image FROM quality_checks where id=?', (id_, ))
        row = cursor.fetchone()

        conn.close()
        return Image.open(io.BytesIO(row['image']))

    def get_last(self, last=11):
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(f'SELECT id, check_time, result, json_data FROM quality_checks ORDER BY id DESC LIMIT ?', (last, ))
        df = _rows_to_dataframe(cursor.fetchall())

        conn.close()

        return df
