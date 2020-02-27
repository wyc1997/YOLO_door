import urllib.request
import os
import argparse
import errno
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from time import time as timer
import csv
import random
import clean_up

# DO NOT USE THIS WHEN FILE AND LABELS ALREADY IN PLACE!!! USE CLEAN_UP.py INSTEAD!!!

# THIS SCRIPT DOWNLOADS DESIRABLE IMAGES IN TO THE SPECIFIED FILE AND THEN ASSIGNS THEM INTO 
# VALIDATION AND TRAINING SET IN YOLO DIRECTORIES. TO ENABLE/DISABLE AUTO-ASSIGNMENT,
# SET ASSIGN TO False.

# start lines first batch 1-1000000 second batch 1000001-2000000 2000001 - 4000000

TRAIN_IMG_CSV = "csv/train-images-boxable-with-rotation.csv"
TRAIN_ANNO_CSV = "csv/train-annotations-bbox.csv"
EXTRACT_CLASS = "Door"
# lines from train-annotations-bbox.csv to read
START_LINE = 1
END_LINE = 4000000
# to download, SET DOWNLOAD TO TRUE
DOWNLOAD = True
DOWNLOAD_DIR = "/media/alfred/data/openimages/train_door"
# TO WRITE LABEL INTO YOLOv3 FORMAT, SET WRITE_LABELS TO TRUE
WRITE_LABELS_PATH = "/media/alfred/data/openimages/train_door_labels"
WRITE_LABELS = True
# TO RANDOMLY ASSIGN INTO TRAIN AND VALID TXT FILES SET ASSIGN TO TRUE
ASSIGN = True
ASSIGN_PATH = "YOLOv3/data/door"
ASSIGN_RATIO = 0.9 # ratio of train images amongst all images

class classList():
    def __init__(self, class_dict):
        self.classes = []
        self.labels = []
        self.num = 0
        for cls in class_dict:
            self.classes.append(cls['name'])
            self.labels.append(cls['label'])
        self.length = len(self.classes)

    def getLabel(self, name):
        if name in self.classes:
            return self.labels[self.classes.index(name)]
        return None

    def getClass(self, label):
        if label in self.labels:
            return self.classes[self.labels.index(label)]
        return None
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.num >= self.length:
            self.num = 0
            raise StopIteration
        ret = {'label': self.labels[self.num], 'name':self.classes[self.num]}
        self.num += 1
        return ret

def download_objects_of_interest(download_list):
    def fetch_url(data):
        try:
            urllib.request.urlretrieve(data[1], f'{DOWNLOAD_DIR}/{data[0]}.jpg')
            return url, None
        except Exception as e:
            return None, e

    start = timer()
    results = ThreadPool(4).imap_unordered(fetch_url, download_list)

    df_pbar = tqdm(total=len(download_list), position=1, desc="Download %: ")

    for url, error in results:
        df_pbar.update(1)
        if error is None:
            pass  # TODO: find a way to do tqdm.write() with a refresh
            # print("{} fetched in {}s".format(url, timer() - start), end='\r')
        else:
            pass  # TODO: find a way to do tqdm.write() with a refresh
            # print("error fetching {}: {}".format(url, error), end='\r')

def tally_downloaded_files(extracted):
    downloaded_files = [f.rstrip(".jpg") for f in os.listdir(DOWNLOAD_DIR) if os.path.isfile(os.path.join(DOWNLOAD_DIR, f))]
    i = 0
    while i < extracted.shape[0]:
        if extracted[i, 0] not in downloaded_files:
            extracted = np.delete(extracted, i, 0)
        else:
            i+=1
    return extracted

def main():
    train_list = np.genfromtxt(TRAIN_IMG_CSV, delimiter=',', usecols=[0, 2], dtype=str)
    print(train_list.shape)
    with open('./csv/class-descriptions-boxable.csv', "r") as cls_file:
        classses_dict = csv.DictReader(cls_file, fieldnames=('label', 'name'))
        classes = classList(classses_dict)
    
    extract_label = classes.getLabel(EXTRACT_CLASS)

    train_ann = np.genfromtxt(TRAIN_ANNO_CSV, dtype=str, delimiter=',', usecols=(0, 2, 4, 5, 6, 7), skip_header=START_LINE, max_rows=END_LINE)
    print(train_ann.shape)
    print(train_ann[:5, ])

    # print(extract_label)
    extract_index = []
    for i in range(train_ann.shape[0]):
        if train_ann[i, 1] == extract_label:
            extract_index.append(i)
    
    print("num extracted: ", len(extract_index))

    extracted = train_ann[extract_index, ]
    print(extracted[:5, ])


    image_list = np.unique(extracted[:, 0])
    print("num unique image:", image_list.shape[0])
    print(image_list[:5, ])

    if DOWNLOAD:
        download_list = []
        for x in image_list:
            try:
                i = np.where(train_list[:,0] == x)
                download_list.append(train_list[i[0][0],].tolist())
            except ValueError:
                print(f'image {x} not found')

        print(len(download_list))
        # print(download_list[1])
        download_objects_of_interest(download_list)
        extracted = tally_downloaded_files(extracted)

    image_list = np.unique(extracted[:, 0])

    if WRITE_LABELS:
        X_coor =  (extracted[:, 2].astype(np.float32) + extracted[:, 3].astype(np.float32)) / 2 # turning max/min to center coor
        Y_coor =  (extracted[:, 4].astype(np.float32) + extracted[:, 5].astype(np.float32)) / 2
        width = extracted[:, 3].astype(np.float32) - extracted[:, 2].astype(np.float32)
        height = extracted[:, 5].astype(np.float32) - extracted[:, 4].astype(np.float32)

        # assign them back to extracted
        extracted[:, 2] = X_coor
        extracted[:, 3] = Y_coor
        extracted[:, 4] = width
        extracted[:, 5] = height

        for i in range(extracted.shape[0]):
            with open(f'{WRITE_LABELS_PATH}/{extracted[i, 0]}.txt', 'a') as label_file:
                label_file.write(f'0 {extracted[i, 2]} {extracted[i, 3]} {extracted[i, 4]} {extracted[i, 5]}\n')
    
    if ASSIGN:
        for i in range(image_list.shape[0]):
            if random.randint(1, 10) < ASSIGN_RATIO * 10:
                with open(f'{ASSIGN_PATH}/train.txt', 'a') as train_file:
                    train_file.write(f'{DOWNLOAD_DIR}/{image_list[i]}.jpg\n') 
            else:
                with open(f'{ASSIGN_PATH}/valid.txt', 'a') as valid_file:
                    valid_file.write(f'{DOWNLOAD_DIR}/{image_list[i]}.jpg\n')



if __name__ == '__main__':
    main()