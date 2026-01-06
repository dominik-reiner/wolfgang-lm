import os
import urllib.request
import zipfile
import shutil
import glob

# Configuration
# specific snapshot from Deutsches Textarchiv (DTA)
# Source: https://www.deutschestextarchiv.de/
# License: CC BY 4.0 (Berlin-Brandenburgische Akademie der Wissenschaften)
DOWNLOAD_URL = "https://www.deutschestextarchiv.de/media/download/dtak/2020-10-23/normalized/Belletristik.zip"
ZIP_FILE = "data/Belletristik_Normalized.zip"
EXTRACT_PATH = "data/temp_extract"
DEST_BELLETRISTIK = "data/Belletristik"
DEST_GESPRAECHE = "data/gespraeche"


def download_file(url, text_file):
    print(f"Downloading {url}...")
    try:
        with urllib.request.urlopen(url) as response, open(text_file, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def setup_directories():
    # Clean up existing directories to ensure a fresh start
    for dir_path in [DEST_BELLETRISTIK, DEST_GESPRAECHE, EXTRACT_PATH]:
        if os.path.exists(dir_path):
            print(f"Clearing {dir_path}...")
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)

    # Ensure ZIP parent folder exists
    os.makedirs(os.path.dirname(ZIP_FILE), exist_ok=True)


def organize_files():
    print("Organizing files...")
    # Find all txt files in the extracted location
    # The zip usually contains a folder named 'Belletristik'
    source_files = glob.glob(os.path.join(EXTRACT_PATH, "**/*.txt"), recursive=True)
    print(f"Found {len(source_files)} files in extracted zip.")

    moved_belletristik = 0
    moved_gespraeche = 0

    for file_path in source_files:
        filename = os.path.basename(file_path)

        # Check if it is an Eckermann file
        if "eckermann" in filename.lower() and "goethe" in filename.lower():
            dest = os.path.join(DEST_GESPRAECHE, filename)
            moved_gespraeche += 1
        else:
            dest = os.path.join(DEST_BELLETRISTIK, filename)
            moved_belletristik += 1

        shutil.move(file_path, dest)

    print(f"Moved {moved_belletristik} files to {DEST_BELLETRISTIK}")
    print(f"Moved {moved_gespraeche} files to {DEST_GESPRAECHE}")


def main():
    setup_directories()

    # 1. Download
    if not os.path.exists(ZIP_FILE):
        if not download_file(DOWNLOAD_URL, ZIP_FILE):
            return
    else:
        print("Zip file already exists, skipping download.")

    # 2. Extract
    print("Extracting...")
    try:
        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid zip file.")
        return

    # 3. Organize
    organize_files()

    # 4. Cleanup
    print("Cleaning up temp files...")
    shutil.rmtree(EXTRACT_PATH)

    print("Data Setup Complete.")


if __name__ == "__main__":
    main()
