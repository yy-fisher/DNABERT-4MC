import torch

import  pandas
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, Dataset
import warnings
from sklearn import preprocessing
import   re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import joblib
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import random
import math
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from filelock import FileLock
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# 改变随机数⽣成器的种⼦，可以在调⽤其他随机模块函数之前调⽤此函数。

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
random_seed(2023)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
