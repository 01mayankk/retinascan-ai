import pandas as pd
import os
import shutil
from pathlib import Path

# =====================================
# PATHS
# =====================================
script_dir = Path(__file__).resolve().parent
csv_file = "../../dataset/train.csv"
image_folder = "../../dataset/raw_images"
output_folder = "../../dataset/classified_images"
csv_path = (script_dir / csv_file).resolve()
image_folder_path = (script_dir / image_folder).resolve()
output_folder_path = (script_dir / output_folder).resolve()

if not csv_path.is_file():
    raise FileNotFoundError(
        f"Expected CSV not found: {csv_path}. "
        "classify_images.py requires dataset/train.csv and dataset/raw_images/."
    )

if not image_folder_path.is_dir():
    raise FileNotFoundError(
        f"Raw image folder not found: {image_folder_path}. "
        "classify_images.py requires the unsplit dataset/raw_images/ input."
    )

# =====================================
# READ CSV
# =====================================
df = pd.read_csv(csv_path)

# =====================================
# CREATE CLASS FOLDERS
# =====================================
for i in range(5):

    os.makedirs(
        os.path.join(output_folder_path, str(i)),
        exist_ok=True
    )

# =====================================
# CLASSIFY IMAGES
# =====================================
count = 0
missing = 0

for _, row in df.iterrows():

    image_name = row['id_code'] + ".png"

    source_path = os.path.join(
        image_folder_path,
        image_name
    )

    diagnosis = str(row['diagnosis'])

    destination_path = os.path.join(
        output_folder_path,
        diagnosis,
        image_name
    )

    if os.path.exists(source_path):

        shutil.copy(source_path, destination_path)

        count += 1

        print(f"✅ Copied: {image_name}")

    else:

        missing += 1

        print(f"❌ Missing: {image_name}")

# =====================================
# FINAL OUTPUT
# =====================================
print("\n🎉 Classification Completed!")
print(f"Copied Images : {count}")
print(f"Missing Images: {missing}")