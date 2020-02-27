import os
import numpy as np
import random
from PIL import Image

CLEAN = True
IMAGE_DIR = "/media/alfred/data/openimages/train_door"
LABEL_DIR = "/media/alfred/data/openimages/train_door_labels"
PARENT_DIR = IMAGE_DIR.rstrip('/train_door')

RESHUFFLE = True
TRAIN_VAL_RATIO = 0.9
TRAIN_FILE = "YOLOv3/data/door/train.txt"
VAL_FILE = "YOLOv3/data/door/valid.txt"

def clean_up_download():
    downloaded_img = [f.rstrip('.jpg') for f in os.listdir(IMAGE_DIR) if os.path.isfile(f'{IMAGE_DIR}/{f}')]
    downloaded_label = [f.rstrip('.txt') for f in os.listdir(LABEL_DIR) if os.path.isfile(f'{LABEL_DIR}/{f}')]

    output = []
    for img in downloaded_img:
        if img in downloaded_label:
            try:
                i = Image.open(f'{IMAGE_DIR}/{img}.jpg')
                i.verify()
            except (IOError, SyntaxError) as e: 
                if os.path.exists(f'{IMAGE_DIR}/{img}.jpg'):
                    os.remove(f'{IMAGE_DIR}/{img}.jpg')
                if os.path.exists(f'{LABEL_DIR}/{img}.txt'):
                    os.remove(f'{LABEL_DIR}/{img}.txt')
            else:
                output.append(img)
    output.sort()

    if os.path.exists(f'{PARENT_DIR}/train_door.txt'):
        os.remove(f'{PARENT_DIR}/train_door.txt')

    with open(f'{PARENT_DIR}/train_door.txt', 'a') as train:
        for img in output:
            train.write(f'{img}\n')

def setup_yolo():
    if not os.path.exists(f'{PARENT_DIR}/train_door.txt'):
        print("train_door.txt file not found")
        return 
    if os.path.exists(TRAIN_FILE):
        os.remove(TRAIN_FILE)
    if os.path.exists(VAL_FILE):
        os.remove(VAL_FILE)

    imgs = []
    with open(f'{PARENT_DIR}/train_door.txt', 'r') as train:
        imgs = [img.rstrip('\n') for img in train.readlines()]
    for img in imgs:
        if random.randint(1, 10) < TRAIN_VAL_RATIO * 10:
            with open(TRAIN_FILE, 'a') as tf:
                tf.write(f'{IMAGE_DIR}/{img}.jpg\n')
        else:
            with open(VAL_FILE, 'a') as vf:
                vf.write(f'{IMAGE_DIR}/{img}.jpg\n')

def main():
    if CLEAN:
        clean_up_download()
    if RESHUFFLE:
        setup_yolo()



if __name__ == "__main__":
    main()