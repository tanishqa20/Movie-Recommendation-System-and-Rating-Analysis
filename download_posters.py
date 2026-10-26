
import os
import time
import pandas as pd
import requests

# =============== CONFIGURATION ===============

METADATA_FILE = "data/movies_metadata.csv"
POSTER_FOLDER = "posters"
OUTPUT_FOLDER = "outputs"
FAILED_FILE = os.path.join(OUTPUT_FOLDER, "failed_posters.csv")

IMAGE_HOSTS = [
    "https://image.tmdb.org/t/p/w500",
    "https://img.amodb.dev/t/p/w500",
]

# Start with True. Change to False after checking the test.
TEST_MODE = True
TEST_LIMIT = 20

# When TEST_MODE is True, test rows 100 to 119.
TEST_START_ROW = 100

PROGRESS_INTERVAL = 100
REQUEST_DELAY = 0.05
TIMEOUT_SECONDS = 12

# =============================================

os.makedirs(POSTER_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if not os.path.isfile(METADATA_FILE):
    raise FileNotFoundError(
        f"Cannot find {METADATA_FILE}. "
        "Run this script from your CineMatch project folder."
    )

print("Loading movie metadata...")
metadata = pd.read_csv(METADATA_FILE, low_memory=False)

required_columns = {"title", "poster_path"}
missing_columns = required_columns - set(metadata.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns in metadata: {sorted(missing_columns)}"
    )

metadata["title"] = metadata["title"].fillna("Unknown").astype(str)
metadata["poster_path"] = (
    metadata["poster_path"].fillna("").astype(str).str.strip()
)

metadata = metadata[
    metadata["poster_path"].str.startswith("/")
].drop_duplicates(subset=["poster_path"]).reset_index(drop=True)

total_available = len(metadata)

if TEST_MODE:
    metadata = metadata.iloc[
        TEST_START_ROW:TEST_START_ROW + TEST_LIMIT
    ].copy()

total = len(metadata)

print(f"Unique poster paths in dataset: {total_available}")
print(f"Posters selected for this run: {total}")
print(f"Test mode: {TEST_MODE}")
print(f"Saving posters to: {os.path.abspath(POSTER_FOLDER)}")

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 CineMatch Movie Recommender"
})

downloaded = 0
skipped = 0
failed = 0
failed_records = []

for number, row in enumerate(metadata.itertuples(index=False), start=1):
    title = row.title.strip()
    poster_path = row.poster_path.strip()
    filename = os.path.basename(poster_path)

    if not filename:
        failed += 1
        failed_records.append({
            "title": title,
            "poster_path": poster_path,
            "status": "INVALID_PATH",
            "reason": "Empty filename",
        })
        continue

    local_path = os.path.join(POSTER_FOLDER, filename)

    # Reuse existing image files.
    if (
        os.path.isfile(local_path)
        and os.path.getsize(local_path) > 1000
    ):
        skipped += 1
        continue

    success = False
    errors = []

    # Try primary host, then alternative host.
    for host in IMAGE_HOSTS:
        try:
            response = session.get(
                host + poster_path,
                timeout=TIMEOUT_SECONDS
            )

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            if (
                response.status_code == 200
                and content_type.startswith("image/")
                and len(response.content) > 1000
            ):
                with open(local_path, "wb") as image_file:
                    image_file.write(response.content)

                downloaded += 1
                success = True
                break

            errors.append(
                f"{host}: HTTP {response.status_code} "
                f"({content_type or 'unknown content type'})"
            )

        except requests.RequestException as error:
            errors.append(f"{host}: {error}")

    if not success:
        failed += 1
        failed_records.append({
            "title": title,
            "poster_path": poster_path,
            "status": "FAILED",
            "reason": " | ".join(errors),
        })

    if (
        number % PROGRESS_INTERVAL == 0
        or number == total
    ):
        print(
            f"Processed {number}/{total} | "
            f"Downloaded: {downloaded} | "
            f"Already available: {skipped} | "
            f"Failed on both hosts: {failed}"
        )

        pd.DataFrame(
            failed_records,
            columns=["title", "poster_path", "status", "reason"]
        ).to_csv(FAILED_FILE, index=False)

    time.sleep(REQUEST_DELAY)

pd.DataFrame(
    failed_records,
    columns=["title", "poster_path", "status", "reason"]
).to_csv(FAILED_FILE, index=False)

print("\n========== DOWNLOAD SUMMARY ==========")
print(f"Processed: {total}")
print(f"New posters downloaded: {downloaded}")
print(f"Existing posters reused: {skipped}")
print(f"Failed on both hosts: {failed}")
print(f"Poster folder: {os.path.abspath(POSTER_FOLDER)}")
print(f"Failure report: {os.path.abspath(FAILED_FILE)}")

if TEST_MODE:
    print("\nTest completed.")
    print("Review the results before enabling full download.")
else:
    print("\nFull download run completed.")
