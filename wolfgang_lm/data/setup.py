import os
import urllib.request
import zipfile
import shutil
import glob

# Configuration
# specific snapshot from Deutsches Textarchiv (DTA)
# Source: https://www.deutschestextarchiv.de/
# License: CC BY 4.0 (Berlin-Brandenburgische Akademie der Wissenschaften)

DATASETS = {
    "Belletristik_Core": "https://www.deutschestextarchiv.de/media/download/dtak/2020-10-23/normalized/Belletristik.zip",
    "Gebrauchsliteratur": "https://www.deutschestextarchiv.de/media/download/dtak/2020-10-23/normalized/Gebrauchsliteratur.zip",
    "Wissenschaft": "https://www.deutschestextarchiv.de/media/download/dtak/2020-10-23/normalized/Wissenschaft.zip",
    "Zeitung": "https://www.deutschestextarchiv.de/media/download/dtae/2020-10-23/normalized/Zeitung.zip",
    "Belletristik_Ext": "https://www.deutschestextarchiv.de/media/download/dtae/2020-10-23/normalized/Belletristik.zip",
}

EXTRACT_PATH = "data/temp_extract"
DEST_DIRS = {
    "Belletristik_Core": "data/Belletristik_Core",
    "Belletristik_Ext": "data/Belletristik_Ext",
    "Gebrauchsliteratur": "data/Gebrauchsliteratur",
    "Wissenschaft": "data/Wissenschaft",
    "Zeitung": "data/Zeitung",
    "gespraeche": "data/gespraeche",
}


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
    for dir_path in list(DEST_DIRS.values()) + [EXTRACT_PATH]:
        if os.path.exists(dir_path):
            print(f"Clearing {dir_path}...")
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)

    # Ensure data directory exists for the zips
    os.makedirs("data", exist_ok=True)


def organize_files(dataset_key):
    """
    Moves files from the extracted folder to the specific destination directory.
    Because DTA zips usually extract to a folder named after the type (e.g. 'Belletristik'),
    we need to map that generic name to our specific target (e.g. 'Belletristik_Core').
    """
    print(f"Organizing {dataset_key}...")

    # Map our dataset keys to the likely folder name inside the zip
    # Examples:
    #   dataset_key="Belletristik_Core" -> zip contains folder "Belletristik"
    #   dataset_key="Zeitung" -> zip contains folder "Zeitung"

    zip_folder_name = dataset_key
    if "Belletristik" in dataset_key:
        zip_folder_name = "Belletristik"

    source_dir = os.path.join(EXTRACT_PATH, zip_folder_name)

    # Fallback: check root if subfolder doesn't exist
    if not os.path.exists(source_dir):
        source_dir = EXTRACT_PATH

    source_files = glob.glob(os.path.join(source_dir, "**/*.txt"), recursive=True)
    print(f"[{dataset_key}] Found {len(source_files)} files.")

    moved_count = 0
    gespraeche_count = 0

    target_dir = DEST_DIRS[dataset_key]
    gespraeche_dir = DEST_DIRS["gespraeche"]

    for file_path in source_files:
        filename = os.path.basename(file_path)

        # Check if it is an Eckermann file
        if "eckermann" in filename.lower() and "goethe" in filename.lower():
            dest = os.path.join(gespraeche_dir, filename)
            gespraeche_count += 1
        else:
            dest = os.path.join(target_dir, filename)
            moved_count += 1

        # Avoid overwriting or errors if moving to same place
        if os.path.abspath(file_path) != os.path.abspath(dest):
            shutil.move(file_path, dest)

    print(f"Moved {moved_count} files to {target_dir}")
    if gespraeche_count > 0:
        print(f"Moved {gespraeche_count} files to {gespraeche_dir}")


def main():
    setup_directories()

    # 1. Download, Extract & Organize
    # We process one by one to avoid collisions in temp_extract (e.g. Belletristik core vs ext)
    for name, url in DATASETS.items():
        zip_path = f"data/{name}.zip"

        print(f"\n--- Processing {name} ---")
        if not os.path.exists(zip_path):
            if not download_file(url, zip_path):
                continue
        else:
            print(f"Zip file {zip_path} already exists, skipping download.")

        # Ensure temp is clean
        if os.path.exists(EXTRACT_PATH):
            shutil.rmtree(EXTRACT_PATH)
        os.makedirs(EXTRACT_PATH, exist_ok=True)

        print(f"Extracting {zip_path}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(EXTRACT_PATH)
        except zipfile.BadZipFile:
            print(f"Error: {zip_path} is not a valid zip file.")
            continue

        # Organize immediately
        organize_files(name)

    # Final Cleanup
    print("Cleaning up temp files...")
    if os.path.exists(EXTRACT_PATH):
        shutil.rmtree(EXTRACT_PATH)

    print("Data Setup Complete.")


if __name__ == "__main__":
    main()
