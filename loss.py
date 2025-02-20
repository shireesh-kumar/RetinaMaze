import torch.nn as nn
import torch
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
  def __init__(self, weight=None, size_average=True):
    super(DiceBCELoss, self).__init__()

  def forward(self, inputs, targets, smooth=1):

    inputs = torch.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    Dice_BCE = BCE + dice_loss

    return Dice_BCE