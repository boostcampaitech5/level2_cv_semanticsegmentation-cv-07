import os 
import random
import pickle
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import albumentations as A
from tqdm.auto import tqdm
import cv2
import torch.nn.functional as F

from model_ad import unet_base


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

SAVED_DIR = "/opt/ml/exp_result/"


### 설정 목록
## model channel(RGB or grey)
## model path
## resize scale

# model = unet_base(is_grey=True)
GREY = True
NLB = False
BATCH_SIZE = 2

model = unet_base(is_grey = GREY, use_nlb= NLB, batch_size= BATCH_SIZE)

## model path 입력
checkpoint = torch.load('/opt/ml/input/code/level2_cv_semanticsegmentation-cv-07/unet_base/result_unet_base_2048_fold4_Woo_focal_Aug/unet_base_2048_fold4_Woo_focal_Aug_best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# model = torch.load(model_path, map_location=device)


# 테스트 데이터 경로를 입력하세요
IMAGE_ROOT = "/opt/ml/input/data/test/DCM"

pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}

RANDOM_SEED = 21

def set_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

# mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다. (mask map -> RLE)
def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# RLE로 인코딩된 결과를 mask map으로 복원합니다. (RLE -> mask map)
def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)



# Dataset for inference
class XRayInferenceDataset(Dataset):
    def __init__(self, transforms=None, grey=False):
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))
        
        self.filenames = _filenames
        self.transforms = transforms
        self.grey = grey
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # print('before:' ,image.shape)
        if self.grey:
            image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2GRAY)
            image = image[..., np.newaxis]  # 1채널로 차원을 추가합니다.
            # print(image.shape)
        
        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        
        image = torch.from_numpy(image).float()
            
        return image, image_name
    
    
def test(model, data_loader, thr=0.5):
    set_seed()
    model = model.cuda()
    model.eval()
        
    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)
        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
                images = images.cuda()    
                outputs = model(images)
            
                # restore original size
                outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
                outputs = torch.sigmoid(outputs.detach().cpu())
            
                outputs = (outputs > thr).detach().cpu().numpy()
            
                for output, image_name in zip(outputs, image_names):
                    for c, segm in enumerate(output):
                        rle = encode_mask_to_rle(segm)
                        rles.append(rle)
                        filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
    
    
    return rles, filename_and_class



tf = A.Resize(2048, 2048)

test_dataset = XRayInferenceDataset(transforms=None, grey =True)


test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=2,
    shuffle=False,
    num_workers=2,
    drop_last=False
)


rles, filename_and_class = test(model, test_loader)

classes, filename = zip(*[x.split("_") for x in filename_and_class])

image_name = [os.path.basename(f) for f in filename]

df = pd.DataFrame({
    "image_name": image_name,
    "class": classes,
    "rle": rles,
})

## csv 파일 이름 입력
df.to_csv("./inference_result/inference_result.csv", index=False)