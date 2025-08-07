import os
import random

# CONFIGURATION
ROOT_DIR = "/home/nvidia10/jetson-inference/python/training/classification/data/Image_Classification_Preprocessed_Dataset/val"
FILES_TO_DELETE_PER_FOLDER = 450

print(f"🔍 Checking directory: {ROOT_DIR}")
print(f"📁 Directory exists: {os.path.exists(ROOT_DIR)}")

if not os.path.exists(ROOT_DIR):
    print("❌ Directory does not exist!")
    exit(1)

print(f"🗂️ Walking through subdirectories...")

# Walk through all subdirectories
subdirs_found = 0
for root, dirs, files in os.walk(ROOT_DIR):
    print(f"📂 Current directory: {root}")
    print(f"   Subdirs: {dirs}")
    print(f"   Files: {len(files)}")
    
    # Skip the root directory itself
    if root == ROOT_DIR:
        print("   ⏭️ Skipping root directory")
        continue

    subdirs_found += 1
    file_paths = [os.path.join(root, f) for f in files]

    if len(file_paths) >= FILES_TO_DELETE_PER_FOLDER:
        files_to_delete = random.sample(file_paths, FILES_TO_DELETE_PER_FOLDER)
        print(f"   🗑️ Deleting {len(files_to_delete)} files from {root}")
        for file_path in files_to_delete:
            os.remove(file_path)
            print(f"     Deleted: {os.path.basename(file_path)}")
    else:
        print(f"   ⏭️ Skipped {root} (only {len(file_paths)} files)")

print(f"\n✅ Processing complete. Found {subdirs_found} subdirectories.")
