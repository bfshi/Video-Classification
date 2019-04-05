from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import glob

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Set(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 2

    def __getitem__(self, item):
        return [np.array([1, 2]), np.array([3, 4])], 'asd', {'a': 1}

train_loader = DataLoader(
        Set(),
        batch_size=2,
    )
for i, item in enumerate(train_loader):
    print(item)

