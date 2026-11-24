import os
from pathlib import Path

# =====================================
# DATASET PATH
# =====================================
script_dir = Path(__file__).resolve().parent
base_folder = "../../dataset/balanced_dataset"
base_folder_path = (script_dir / base_folder).resolve()

if not base_folder_path.is_dir():
    raise FileNotFoundError(
        f"Balanced dataset folder not found: {base_folder_path}. "
        "This counter reads the output from balance_dataset.py."
    )

print("\n📊 Balanced Dataset Count\n")

total = 0

# =====================================
# COUNT IMAGES
# =====================================
for folder in sorted(os.listdir(base_folder_path)):

    folder_path = os.path.join(
        base_folder_path,
        folder
    )

    if os.path.isdir(folder_path):

        count = len([

            file for file in os.listdir(folder_path)

            if file.endswith(
                ('.png', '.jpg', '.jpeg')
            )
        ])

        total += count

        print(f"Class {folder}: {count} images")

# =====================================
# TOTAL
# =====================================
print(f"\nTotal Images: {total}")