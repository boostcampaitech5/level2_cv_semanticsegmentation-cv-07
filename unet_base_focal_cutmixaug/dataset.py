# python native
import os
import json
import random
import pandas as pd
# external library
import cv2
import numpy as np
from sklearn.model_selection import GroupKFold
import albumentations as A

# torch
import torch
from torch.utils.data import Dataset

# visualization
import matplotlib.pyplot as plt


# 데이터 경로를 입력하세요
IMAGE_ROOT = "/opt/ml/input/data/train/DCM"
LABEL_ROOT = "/opt/ml/input/data/train/outputs_json"
AUX_ROOT = '/opt/ml/input/data'

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}

jsons = {
    os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
    for root, _dirs, files in os.walk(LABEL_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".json"
}

jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

pngs = sorted(pngs)
jsons = sorted(jsons)

class XRayDataset(Dataset):
    def __init__(self, is_train=True, transforms=None, fold=0, grey=False, cutmix=False):
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        
        self.grey = grey
        
        # split train-valid
        # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
        # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
        groups = [os.path.dirname(fname) for fname in _filenames]
        
        # dummy label
        ys = [0 for fname in _filenames]
        
        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)
        
        train_filenames = []
        train_labelnames = []
        valid_filenames = []
        valid_labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            # 0번을 validation dataset으로 사용합니다.
            if i == fold:
                valid_filenames += list(_filenames[y])
                valid_labelnames += list(_labelnames[y])

            else:
                train_filenames += list(_filenames[y])
                train_labelnames += list(_labelnames[y])
        
        self.is_train = is_train
        self.cutmix = cutmix
        self.transforms = transforms
        if self.is_train:
            self.filenames = train_filenames
            self.labelnames = train_labelnames
        else:
            self.filenames = valid_filenames
            self.labelnames = valid_labelnames
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)
        
        # process a label of shape (H, W, NC)
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # iterate each class
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon to mask
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        if self.is_train and self.cutmix:
            image, label = cutmix(image,label)
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label
            
        
        # print('before:' ,image.shape)
        if self.grey:
            image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2GRAY)
            image = image[..., np.newaxis]  # 1채널로 차원을 추가합니다.
            # print(image.shape)

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        label = label.transpose(2, 0, 1)
        
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label

def cutmix(img,label,p=0.2):
    
    if p < random.uniform(0,1):
        return img, label

    #손등 부분 crop
    tf_cutmix = A.Compose([
    A.Crop(x_min=750, y_min=1250, x_max=1250, y_max=1750,p=1)
        ])

    transform = tf_cutmix(image=img, mask=label)

    #아래 부분 양 옆 이미지 합성하기
    img[2048-500:,0:500] = transform['image']
    img[2048-500:,2048-500:] = transform['image']
    label[2048-500:,0:500] = transform['mask']
    label[2048-500:,2048-500:] = transform['mask']

    return img, label

class XRayAuxDataset(Dataset):
    def __init__(self, is_train=True, transforms=None, fold=0, grey=False, cutmix=False):
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        
        self.grey = grey
        
        # split train-valid
        # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
        # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
        groups = [os.path.dirname(fname) for fname in _filenames]
        
        # dummy label
        ys = [0 for fname in _filenames]
        
        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)
        
        train_filenames = []
        train_labelnames = []
        valid_filenames = []
        valid_labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            # 0번을 validation dataset으로 사용합니다.
            if i == fold:
                valid_filenames += list(_filenames[y])
                valid_labelnames += list(_labelnames[y])

            else:
                train_filenames += list(_filenames[y])
                train_labelnames += list(_labelnames[y])

        #aux
        self.aux_df = pd.read_csv('/opt/ml/input/data/meta_data.csv')
        self.aux_df.drop('Unnamed: 0',axis=1, inplace=True)
        self.aux_data = self.aux_df.values

        
        self.is_train = is_train
        self.cutmix = cutmix
        self.transforms = transforms
        if self.is_train:
            self.filenames = train_filenames
            self.labelnames = train_labelnames
        else:
            self.filenames = valid_filenames
            self.labelnames = valid_labelnames
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_id = int(image_name.split('/')[0][2:])
        image_path = os.path.join(IMAGE_ROOT, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)
        
        # process a label of shape (H, W, NC)
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # iterate each class
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon to mask
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        if self.is_train and self.cutmix:
            image, label = cutmix(image,label)
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label
            
        
        # print('before:' ,image.shape)
        if self.grey:
            image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2GRAY)
            image = image[..., np.newaxis]  # 1채널로 차원을 추가합니다.
            # print(image.shape)

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        label = label.transpose(2, 0, 1)
        
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        aux = torch.from_numpy(self.aux_data[image_id-1]).float()    
        return image, label, aux