import os
import shutil

# Root directory for your workspace
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Walk through all directories and delete __pycache__ folders
for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
    for dirname in dirnames:
        if dirname == "__pycache__":
            pycache_path = os.path.join(dirpath, dirname)
            print(f"Deleting: {pycache_path}")
            shutil.rmtree(pycache_path)
print("All __pycache__ folders deleted.")
