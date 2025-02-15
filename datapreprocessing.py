from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import config
import pytorch_lightning as pl
from glob import glob
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class FIVESDataset(Dataset):
  def __init__(self, images_path, masks_path):
    self.images_path = images_path
    self.masks_path = masks_path
    self.n_samples = len(images_path)

  def __getitem__(self, index):
    image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
    image = cv2.resize(image, (config.H, config.W), interpolation=cv2.INTER_AREA)
    image = image/255.0 ## ex.(256,256, 3)
    image = np.transpose(image, (2, 0, 1))  ## ex.(3, 256, 256)
    image = image.astype(np.float32)
    image = torch.from_numpy(image)

    mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (config.H, config.W), interpolation=cv2.INTER_NEAREST)
    mask = mask/255.0   ## ex.(256, 256)
    mask = np.expand_dims(mask, axis=0) ## ex.(1, 256, 256)
    mask = mask.astype(np.float32)
    mask = torch.from_numpy(mask)

    return image, mask

  def __len__(self):
    return self.n_samples
  
  
class RVDataModule(pl.LightningDataModule):
  def __init__(self, data_dir, batch_size, num_workers):
    super().__init__()
    self.data_dir = data_dir 
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.train_x = sorted(glob(config.TRAIN_ORIGINAL_PATH + '/*.png'))
    self.train_y = sorted(glob(config.TRAIN_GROUND_PATH + '/*.png'))

  def prepare_data(self):
    self.train_x, self.valid_x, self.train_y, self.valid_y = train_test_split(self.train_x, self.train_y, test_size=0.2, random_state=42)
    
  def setup(self, stage):
    self.train_dataset = FIVESDataset(self.train_x, self.train_y)
    self.valid_dataset = FIVESDataset(self.valid_x, self.valid_y)

  def train_dataloader(self):
    return DataLoader(
      dataset=self.train_dataset,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=self.num_workers)
    
  def val_dataloader(self):
    return DataLoader(
      dataset=self.valid_dataset,
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=4,)
    
    


