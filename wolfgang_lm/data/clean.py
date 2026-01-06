import os
import re
import glob
from pathlib import Path

# Configuration
SOURCE_DIR = "data"
DEST_DIR = "data_clean"
BELLETRISTIK_DIR = os.path.join(SOURCE_DIR, "Belletristik")
GESPRAECHE_DIR = os.path.join(SOURCE_DIR, "gespraeche")


def clean_artifacts(text):
    """Remove OCR artifacts, page numbers, and headers."""
    lines = text.splitlines()
    cleaned_lines = []

    # Heuristics for header/footer removal
    # DTA texts often have a standard header block. We might look for the start of the actual content.
    # For now, we'll try a generic line-by-line cleaner.

    for line in lines:
        line = line.strip()
        if not line:
            cleaned_lines.append("")
            continue

        # Remove page markers like [0012] or [[1]/0011]
        # Regex: \[.*?\] might be too aggressive if text uses brackets.
        # Specific pattern for these files seems to be `[1234]`, `[0012]`, `[[1]/...`
        if re.match(r"^\[.*?\d+.*?\]$", line):
            continue

        # Remove form feed characters
        if "\x0c" in line:
            line = line.replace("\x0c", "")

        # Remove likely page numbers (single numbers on a line)
        if line.isdigit() and len(line) < 4:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def process_file(filepath, dest_subdir):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        content = clean_artifacts(content)

        # Ensure destination directory exists
        dest_path = Path(DEST_DIR) / dest_subdir / os.path.basename(filepath)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(content)

        return True
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    print("Starting Preprocessing...")

    # Process Belletristik
    belletristik_files = glob.glob(os.path.join(BELLETRISTIK_DIR, "*.txt"))
    print(f"Found {len(belletristik_files)} Belletristik files.")

    for f in belletristik_files:
        process_file(f, "Belletristik")

    # Process Gespraeche
    gespraeche_files = glob.glob(os.path.join(GESPRAECHE_DIR, "*.txt"))
    print(f"Found {len(gespraeche_files)} Gespraeche files.")

    for f in gespraeche_files:
        process_file(f, "gespraeche")

    print("Preprocessing complete. Output in 'data_clean/'.")


if __name__ == "__main__":
    main()
