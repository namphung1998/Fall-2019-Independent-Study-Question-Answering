"""
This module contains layers to be used in the BiDAF
(Bi-directional Attention Flow) for question answering
(Seo et al, 2016)

Author: Nam Phung
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class HighwayNetwork(nn.Module):
    """
    Highway network layer (Srivastava et al, 2015)
    This network applies a transformation function, as well as
    carries over parts of the input through a gate

    Code adapted from https://github.com/chrischute/squad/blob/master/layers.py
    """

    def __init__(self, num_layers, input_size):
        """
        Initializes layers
        :param num_layers: (int) Number of layers
        :param input_size: (int) Size of input tensor
        """
        super(HighwayNetwork, self).__init__()
        self.transforms = nn.ModuleList([
            nn.Linear(input_size, input_size) for _ in range(num_layers)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_size, input_size) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        Single forward pass of highway network
        :param x: (Tensor) tensor of shape (batch_size, seq_len, input_size)
        :return: x (Tensor) tensor of shape (batch_size, seq_len, input_size)
        """
        for transform, gate in zip(self.transforms, self.gates):
            t = F.sigmoid(gate(x))  # shape (batch_size, seq_len, input_size)
            h = F.relu(transform(x))  # shape (batch_size, seq_len, input_size)
            x = t * h + (1 - t) * x

        return x


class WordEmbedding(nn.Module):
    """
    Word embedding layer for BiDAF, uses pretrained GloVe embeddings.
    The embeddings are then fine-tuned using a 2-layer Highway Network

    """

    def __init__(self, embeddings, hidden_size, drop_prob=0.0):
        """
        Initializes layers
        :param embeddings: (Tensor) pretrained embeddings
        :param hidden_size: (int) hidden size of highway network
        :param drop_prob: (float) dropout probability
        """
        super(WordEmbedding, self).__init__()
        self.drop_prob = drop_prob
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        self.highway_proj = nn.Linear(embeddings.shape[1], hidden_size)
        self.highway = HighwayNetwork(2, hidden_size)

    def forward(self, x):
        """
        Single forward pass of embedding layer
        :param x: (Tensor) tensor of shape (batch_size, seq_len) containing the word indices
        :return: embedded (Tensor)
        """

        embedded = self.embedding(x)
        embedded = self.highway_proj(embedded)
        embedded = F.dropout(embedded, self.drop_prob, self.training)
        embedded = self.highway(embedded)

        return embedded


class Encoder(nn.Module):
    """
    An RNN for encoding a sequence. The output of this layer is the RNN's hidden state at each timestep
    """

    def __init__(self, input_size, hidden_size, num_layers, drop_prob=0.0):
        """
        Initializes the layer
        :param input_size: (int) 
        :param hidden_size: (int)
        :param num_layers: (int) 
        :param drop_prob: (float) Dropout probability
        """

        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.0 if num_layers == 1 else drop_prob)
        self.drop_prob = drop_prob

    def forward(self, x, lengths):
        """
        Single forward pass
        :param x: (Tensor) input tensor of shape (batch_size, seq_len, input_size)
        :param lengths: (LongTensor) lengths of all sequences in the batch
        :return: enc_hiddens (Tensor) hidden state at each timestep
        """

        orig_length = x.shape[1]

        lengths, sorted_idx = torch.sort(lengths, dim=0, descending=True)
        x = x[sorted_idx]
        x = pack_padded_sequence(x, lengths)

        self.lstm.flatten_parameters()  # for suppressing warnings on CUDA device

        enc_hiddens, _ = self.lstm(x)  # shape (batch_size, seq_len, 2 * hidden_size)
        enc_hiddens = pad_packed_sequence(enc_hiddens, batch_first=True, total_length=orig_length)
        _, unsorted_idx = torch.sort(sorted_idx, dim=0)
        enc_hiddens = enc_hiddens[unsorted_idx]
        enc_hiddens = F.dropout(enc_hiddens, self.drop_prob, self.training)

        return enc_hiddens


class Attention(nn.Module):
    """
    Bidirectional Attention Flow layer
    """
