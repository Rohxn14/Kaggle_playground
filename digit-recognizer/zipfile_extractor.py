import zipfile
import os

zip_file_path = "C:\Users\Rohan\Desktop\Kaggle_playground\digit-recognizer\digit-recognizer.zip"
extract_to_path = "C:\Users\Rohan\Desktop\Kaggle_playground\digit-recognizer"

os.makedirs(extract_to_path, exist_ok = True)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)

print(f"Extracted all files to{extract_to_path}")