import os
import cv2
import shutil
import glob
from tqdm import tqdm

file_path = 'save_video'

def remove_empty_dirs(root: str, remove_root=False) -> list[str]:
    removed = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        if not os.listdir(dirpath):
            print(f"🧹 delete: {dirpath}")
            os.rmdir(dirpath)
            removed.append(dirpath)
    if remove_root and os.path.exists(root) and not os.listdir(root):
        os.rmdir(root)
        removed.append(root)
    return removed

remove_empty_dirs(file_path, True)
