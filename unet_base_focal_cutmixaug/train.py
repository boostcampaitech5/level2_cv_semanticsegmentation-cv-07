# python native
import os
import random
from datetime import datetime, timedelta
import importlib


# external library
import numpy as np
from tqdm.auto import tqdm
import albumentations as A
import wandb
import argparse
import json

# torch
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

# visualization
import matplotlib.pyplot as plt

# model, dataset
from dataset import *
from loss import *
from model_ad import unet_base, Aux_UNet

############## PARSE ARGUMENT ########################
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--load_from", type=bool, default=True)
    parser.add_argument("--run_id", type=str)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--seed", type=int,default=21)

    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--loss", type=str, default='focal_dice_loss')     ## default bce loss
    
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr",  type=float, default=0.001)
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--grey", type=bool, default=True)         ## using grey scale image(dataset, model)
    parser.add_argument("--nlb", type=bool, default=False)
    parser.add_argument("--cutmix", type=bool, default=True)## Use non local block or not
    parser.add_argument("--accum", type=int, default=1)             ## gradient accumulation (BATCH = batch_size * accum)
    parser.add_argument("--amp", type=bool, default=True)           ## Use Auto mixed precision or not
    parser.add_argument("--aux", type=bool, default=False)
    args = parser.parse_args()
    return args

args = parse_args()

############## TRAINING SETTINGS 1 ########################
MODEL = 'unet_base_2048_fold4_Woo_AUX'                    ## exp_name
NAME = MODEL
LOAD_FROM = args.load_from
BATCH_SIZE = args.batch_size
LR = args.lr
RANDOM_SEED = args.seed
NUM_EPOCHS = args.epochs
SIZE = args.resolution
FOLD = args.fold
VAL_EVERY = 1
SAVED_DIR = f"./result_{MODEL}/"
GREY = args.grey
NLB = args.nlb
CUTMIX = args.cutmix
ACCUM = args.accum
AMP = args.amp
AUX = args.aux

### GET LOSS FUNCTION
loss_module = importlib.import_module('loss')
CRITERION = getattr(loss_module, args.loss)
####################

if not os.path.isdir(SAVED_DIR):                                                           
    os.mkdir(SAVED_DIR)

############## WANDB RESUME ##########################
if args.resume == True:
    if args.run_id == None:
        print("You didn't type the wandb run id! Please insert wandb run id and try again!")
        raise SystemExit(0)
    dict_run_id = {"run_id": args.run_id}
    json_data = json.dumps(dict_run_id)
    with open('./wandb/wandb-resume.json', 'w') as f:
        f.write(json_data)
   
    run = wandb.init(project="Segmentation", entity="oif", name=NAME, resume=True)
else:
    run = wandb.init(project="Segmentation", entity="oif", name=NAME, resume=False)

############## DATASET SETTINGS & DATA AUGMENTATION ######################
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

train_tf = A.Compose([
     # A.CropNonEmptyMaskIfExists(1024,1024,p=0.5),
     A.Rotate(p=0.2),
     A.HorizontalFlip(p=0.2),
     A.VerticalFlip(p=0.2),
     A.ElasticTransform(p=0.2),
     A.Sharpen()
    ])

train_dataset = XRayDataset(is_train=True, transforms=train_tf, fold=FOLD, grey=GREY, cutmix=CUTMIX)
valid_dataset = XRayDataset(is_train=False, transforms=None, fold=FOLD, grey=GREY)


if AUX:
    train_dataset = XRayAuxDataset(is_train=True, transforms=train_tf, fold=FOLD, grey=GREY, cutmix=CUTMIX)


train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=1,
    drop_last=True,
)

valid_loader = DataLoader(
    dataset=valid_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    drop_last=False
)

############## HELPER METHODS ########################

def save_model(model, file_name=f'{MODEL}_best_model.pt'):
    output_path = os.path.join(SAVED_DIR, file_name)
    torch.save(model, output_path)

def save_model2(model, epoch, optimizer, best_dice, file_name=f'{MODEL}_best_model.pt'):
    output_path = os.path.join(SAVED_DIR, file_name)
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                }, output_path)
    
def set_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
def validation(epoch, model, data_loader, criterion, AUX, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            if AUX:
                outputs, aux = model(images)
            else:
                outputs = model(images)
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # restore original size
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")

            loss = criterion(pred=outputs, target=masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()
            
            dice = dice_coef(outputs, masks)
            dices.append(dice)
            
            # if step == 0:  # Log only for the first batch of images
            #     wandb.log({"Example Images": [wandb.Image(image) for image in images]})
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    avg_dice = torch.mean(dices_per_class).item()
    
    # Log dice scores
    wandb.log({f"Dice Score/{c}": d.item() for c, d in zip(CLASSES, dices_per_class)})
    wandb.log({"Average Dice Score": avg_dice})
    
    return avg_dice

def train(model, data_loader, val_loader, criterion, optimizer, scheduler, AUX):
    # torch.autograd.set_detect_anomaly(True)
    n_class = len(CLASSES)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    best_dice = 0.
    starting_epoch = 0
    
    if wandb.run.resumed:
        checkpoint = torch.load(os.path.join(SAVED_DIR, f'{MODEL}_best_model.pt'))   
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint['best_dice']
        print(f"Resuming Training From Epoch {starting_epoch}...")
    else:
        print(f'Start training..')
    
    for epoch in range(starting_epoch, NUM_EPOCHS):
        model.train()

        for step, (images, masks) in enumerate(data_loader):            
            # gpu 연산을 위해 device 할당
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()

            with torch.cuda.amp.autocast(enabled=AMP):
                outputs = model(images)
                loss = criterion(pred=outputs, target=masks)

            loss /= ACCUM                           ## gradient accumulation
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1) ## gradient clipping
            
            if (step + 1) % ACCUM ==0:      ## loss update(accumulation)
                scaler.step(optimizer)
                scaler.update()

            # step 주기에 따른 loss 출력
            if (step + 1) % 10 == 0:
                current_time = (datetime.now() + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f'{current_time} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
                # Log the loss
                wandb.log({"Loss": loss.item()})

        scheduler.step()
             
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader, criterion, AUX)
            print(f'Val Dice: {dice:.4f}')
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice
                save_model2(model, epoch=epoch, optimizer=optimizer, best_dice=best_dice)

                # Save the best model
                wandb.run.summary["Best Dice Score"] = dice
            else:
                save_model2(model, epoch=epoch, optimizer=optimizer, best_dice=best_dice, file_name="latest.pt")
    print("Training is Finished!")


def aux_train(model, data_loader, val_loader, seg_criterion, aux_criterion, optimizer, scheduler, AUX):
    # torch.autograd.set_detect_anomaly(True)
    n_class = len(CLASSES)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    best_dice = 0.
    starting_epoch = 0
    
    if wandb.run.resumed:
        checkpoint = torch.load(os.path.join(SAVED_DIR, f'{MODEL}_best_model.pt'))   
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint['best_dice']
        print(f"Resuming Training From Epoch {starting_epoch}...")
    else:
        print(f'Start training..')
    
    for epoch in range(starting_epoch, NUM_EPOCHS):
        model.train()

        for step, (images, masks, aux) in enumerate(data_loader):            
            # gpu 연산을 위해 device 할당
            images, masks, auxs = images.cuda(), masks.cuda(), aux.cuda()
            model = model.cuda()

            with torch.cuda.amp.autocast(enabled=AMP):
                seg_outputs, aux_outputs = model(images)
                seg_loss = seg_criterion(seg_outputs, masks)
                aux_loss = aux_criterion(aux_outputs, auxs)
                loss = seg_loss + aux_loss
                
            loss /= ACCUM                           ## gradient accumulation
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1) ## gradient clipping
            
            if (step + 1) % ACCUM ==0:      ## loss update(accumulation)
                scaler.step(optimizer)
                scaler.update()

            # step 주기에 따른 loss 출력
            if (step + 1) % 10 == 0:
                current_time = (datetime.now() + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f'{current_time} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}',
                    f'Seg_Loss: {round(seg_loss.item(),4)}',
                    f'Aux_Loss: {round(aux_loss.item(),4)}',
                )
                # Log the loss
                wandb.log({"Loss": loss.item(), "Aux_Loss": aux_loss.item()})

        scheduler.step()
             
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader, seg_criterion, AUX)
            print(f'Val Dice: {dice:.4f}')
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice
                save_model2(model, epoch=epoch, optimizer=optimizer, best_dice=best_dice)

                # Save the best model
                wandb.run.summary["Best Dice Score"] = dice
            else:
                save_model2(model, epoch=epoch, optimizer=optimizer, best_dice=best_dice, file_name="latest.pt")
    print("Training is Finished!")
    
############## TRAINING SETTINGS 2######################
# Model
model = Aux_UNet()

if LOAD_FROM:
    LOAD_MODEL = 'unet_base_2048_fold4_Woo_focal_Aug'
    LOAD_DIR = os.path.join("/opt/ml/input/code/level2_cv_semanticsegmentation-cv-07/unet_base", f'result_{LOAD_MODEL}')
    checkpoint = torch.load(os.path.join(LOAD_DIR, f"{LOAD_MODEL}_best_model.pt"))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f'Load Weight from {LOAD_MODEL}_best_model.pt')
    

# Optimizer
optimizer = optim.AdamW(params=model.parameters(), lr=LR)

# Scheduler
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch, verbose=True)

# Set Seed
set_seed()

if AUX:
    aux_criterion = nn.MSELoss()
    aux_train(model, train_loader, valid_loader, CRITERION, aux_criterion, optimizer, scheduler, AUX)
else:
    train(model, train_loader, valid_loader, CRITERION, optimizer, scheduler)

if __name__ == '__main__':
    args = parse_args()