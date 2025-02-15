import os
import numpy as np
import torch
import random

def seeding(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)