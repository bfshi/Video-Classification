from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


def getBackbone():
    """
    build a pre-trained resnet101 backbone which
    returns features from its penultimate layer.

    :return: pre-trained resnet101
    """