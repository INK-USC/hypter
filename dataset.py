import os
import json
import re
import string
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from my_datasets import ZSREData, ZESTData, ZSREGroupedData

def MyDatasetCollection(logger, args, data_path, is_training):
    if args.dataset == 'zsre':
        return ZSREData(logger, args, data_path, is_training)
    elif args.dataset == 'zsre_grouped':
        return ZSREGroupedData(logger, args, data_path, is_training)
    elif args.dataset == 'zest':
        return ZESTData(logger, args, data_path, is_training)
    else:
        raise NotImplementedError
