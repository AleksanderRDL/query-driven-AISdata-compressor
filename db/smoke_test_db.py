import os

import psycopg
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    db_url = os.environ["DATABASE_URL"]

    with psycopg.connect(db_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT postgis_version();")
        postgis_row = cur.fetchone()
        if postgis_row is None:
            raise RuntimeError("PostGIS version query returned no row.")
        print("PostGIS:", postgis_row[0])

        cur.execute("SELECT COUNT(*) FROM ais_points_cleaned;")
        count_row = cur.fetchone()
        if count_row is None:
            raise RuntimeError("ais_points_cleaned count query returned no row.")
        print("Rows in ais_points_cleaned:", count_row[0])


if __name__ == "__main__":
    main()
