from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


def getLSTM():
    """
    build a LSTM with a global memory

    :return: a LSTM with a global memory
    """