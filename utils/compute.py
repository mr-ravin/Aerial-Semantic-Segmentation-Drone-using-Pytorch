import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, smooth=1e-6, lambda_focal=0.25, lambda_dice=0.25, Lambda_bce=0.5):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCELoss()
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        self.lambda_bce = Lambda_bce

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        focal_loss = self.focal_loss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        return self.lambda_focal * focal_loss + self.lambda_dice * dice_loss +self.lambda_bce * bce_loss


def get_iou_score(inputs, targets):
    SMOOTH = 1e-6
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    inputs = (inputs > 0.5).int()
    targets = (targets > 0.5).int()
    intersection = (inputs * targets).sum()
    union = inputs.sum() + targets.sum()
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou
