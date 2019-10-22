"""
This module contains all the layers to be used in this
experimental model.

Author: Nam Phung
"""

import torch
import torch.nn as nn

from util import masked_softmax


class CharCNNEmbedding(nn.Module):
    """
    Character-level embeddings. Uses a 1-d convolutional network
    """
    def __init__(self, input_size, embed_size, filters, filter_width):
        super(CharCNNEmbedding, self).__init__()
        self.conv = nn.Conv1d(input_size, )

    def forward(self, x):
        pass
