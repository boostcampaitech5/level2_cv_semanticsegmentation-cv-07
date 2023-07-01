# torch
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def dice_loss(pred=None, target=None, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()   
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def bce_dice_loss(pred=None, target=None, bce_weight = 0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss

def focal_dice_loss(pred=None, target=None, focal_weight = 0.5):
    focal = focal_loss(pred, target)
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = focal * focal_weight + dice * (1 - focal_weight)
    return loss

def focal_loss(inputs, targets, alpha=.25, gamma=2) : 
    BCE = F.binary_cross_entropy_with_logits(inputs, targets)
    BCE_EXP = torch.exp(-BCE)
    loss = alpha * (1-BCE_EXP)**gamma * BCE
    return loss

def bce_loss(pred=None, target=None):
    # Clipping logits to 0
    # pred = torch.clamp(pred, min=-20)
    loss = nn.BCEWithLogitsLoss()(pred, target)    ## nan
    # loss = nn.BCELoss()(torch.sigmoid(pred), target)    ## to avoid nan value
    # loss = nn.BCELoss()(pred.sigmoid(), target)    ## to avoid nan value

    return loss