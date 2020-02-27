import numpy as np
import csv
import cv2
import random

DATA_PATH = '/media/alfred/data/openimages/validation'
EXTRACT_CLASS = 'Door'
IMG_WRITE_PATH = '/media/alfred/data/openimages/val-door'
LABEL_WRITE_PATH = './YOLOv3/data/door/labels'

class classList():
    def __init__(self, class_dict):
        self.classes = []
        self.labels = []
        self.num = 0
        for cls in class_dict:
            self.classes.append(cls['name'])
            self.labels.append(cls['label'])
        self.length = len(self.classes)
        print(self.length)

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

def main():
    with open('./csv/class-descriptions-boxable.csv', "r") as cls_file:
        classses_dict = csv.DictReader(cls_file, fieldnames=('label', 'name'))
        classes = classList(classses_dict)
        # for cls in classes:
        #     print(cls['name'])
            
    extract_label = classes.getLabel(EXTRACT_CLASS)
    
    with open('./csv/validation-annotations-bbox.csv', "r") as ann_file:
        annotations = [ann for ann in csv.DictReader(ann_file)]
    extracted = []
    for ann in annotations:
        if ann['LabelName'] == extract_label:
            extracted.append(ann)
    print(len(extracted))
    # print(extracted)

# displaying images and bboxes
    # index = 103
    # imgdata = extracted[index]
    # imgID = extracted[index]['ImageID']
    # print(imgID)
    # img = cv2.imread(f'{DATA_PATH}/{imgID}.jpg')

    # pt1 = (int(float(imgdata['XMin'])*img.shape[1]), int(float(imgdata['YMax'])*img.shape[0]))
    # pt2 = (int(float(imgdata['XMax'])*img.shape[1]), int(float(imgdata['YMin'])*img.shape[0]))

    # cv2.rectangle(img, pt1, pt2, (0,255,0),3)
    # # cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
    # cv2.imshow('door', img)
    # cv2.waitKey(10000)
    # cv2.destroyAllWindows()

# write all found images 
    # imgIDs = []
    # for imgdata in extracted:
    #     if not imgdata['ImageID'] in imgIDs:
    #         imgIDs.append(imgdata['ImageID'])
    # print(len(imgIDs))
    # for imgID in imgIDs:
    #     img = cv2.imread(f'{DATA_PATH}/{imgID}.jpg')

    #     cv2.imwrite(f'{WRITE_PATH}/{imgID}.jpg', img)

# adapting to Pytorch YOLOv3
    imgIDs = []
    for imgdata in extracted:
        if not imgdata['ImageID'] in imgIDs:
            imgIDs.append(imgdata['ImageID'])
        # X_coor = (float(imgdata['XMax']) + float(imgdata['XMin'])) / 2 # turning max/min to center coor
        # Y_coor = (float(imgdata['YMax']) + float(imgdata['YMin'])) / 2

        # width = float(imgdata['XMax']) - float(imgdata['XMin'])
        # height = float(imgdata['YMax']) - float(imgdata['YMin'])
        # imgID = imgdata['ImageID']
        # with open(f'{LABEL_WRITE_PATH}/{imgID}.txt', 'a') as label_file: #write to label files
        #     label_file.write(f'0 {X_coor} {Y_coor} {width} {height}\n')


    # write to valid.txt/train.txt to indicate img to be validation or training 
    for imgID in imgIDs:
        if random.randint(0, 9) > 8:
            with open('./YOLOv3/data/door/valid.txt', 'a') as valid_file:
                valid_file.write(f'data/door/images/{imgID}.jpg\n')
        else:
            with open('./YOLOv3/data/door/train.txt', 'a') as train_file:
                train_file.write(f'data/door/images/{imgID}.jpg\n')
    # write all found images
    #     img = cv2.imread(f'{DATA_PATH}/{imgID}.jpg')

    #     cv2.imwrite(f'{IMG_WRITE_PATH}/{imgID}.jpg', img)


    


if __name__ == "__main__":
    main()    