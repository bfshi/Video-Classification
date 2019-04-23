from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import _init_paths
from core.config import config


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTMCell(config.MODEL.LSTM_INDIM, config.MODEL.LSTM_OUTDIM)
        self.h = None
        self.c = None
        #TODO: global memory

    def reset(self, batch_size, h_c = None):
        if h_c == None:
            self.h = torch.zeros((batch_size, config.MODEL.LSTM_OUTDIM)).cuda()
            self.c = torch.zeros((batch_size, config.MODEL.LSTM_OUTDIM)).cuda()
        else:
            self.h = h_c[0]
            self.c = h_c[1]

    def forward(self, input):
        self.h = self.h.detach()
        self.c = self.c.detach()
        self.h, self.c = self.lstm(input, (self.h, self.c))
        return self.h, self.c
        # h, c = self.lstm(input, (self.h, self.c))
        # return h, c




def getLSTM():
    """
    build a LSTM with a global memory

    :return: a LSTM with a global memory
    """
    model = LSTM()

    return model
