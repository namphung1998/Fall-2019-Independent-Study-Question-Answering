"""
This module contains layers to be used in the BiDAF
(Bi-directional Attention Flow) for question answering
model (Seo et al, 2016)

Author: Nam Phung
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from util import masked_softmax


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
            t = torch.sigmoid(gate(x))  # shape (batch_size, seq_len, input_size)
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

        orig_len = x.shape[1]

        lengths, sort_idx = torch.sort(lengths, dim=0, descending=True)
        x = x[sort_idx]
        x = pack_padded_sequence(x, lengths, batch_first=True)

        self.lstm.flatten_parameters()
        enc_hiddens, (last_hidden, last_cell) = self.lstm(x)  # enc_hiddens is a PackedSequence object
        # last_hidden is of shape (num_layers * num_directions, batch_size, hidden_size)
        enc_hiddens, _ = pad_packed_sequence(enc_hiddens, batch_first=True, total_length=orig_len)
        _, unsort_idx = torch.sort(sort_idx, dim=0)
        enc_hiddens = enc_hiddens[unsort_idx]  # enc_hiddens is now a Tensor of shape (batch_size, seq_len, 2 * hidden_size)

        return enc_hiddens


class Attention(nn.Module):
    """
    Bidirectional Attention Flow layer

    Code adapted from https://github.com/chrischute/squad/blob/master/layers.py
    """

    def __init__(self, hidden_size, drop_prob=0.0):
        super(Attention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1), requires_grad=True)
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1), requires_grad=True)
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)

        for w in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(w)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, context, question, c_masks, q_masks):
        """
        Single forward pass of attention layer
        :param context: (Tensor) tensor of shape (batch, c_len, hidden_size)
        :param question: (Tensor) tensor of shape (batch, q_len, hidden_size)
        :param c_masks:
        :param q_masks:
        :return:
        """

        batch_size, c_len, _ = context.shape
        q_len = question.shape[1]
        s = self.get_similarity_matrix(context, question) # shape (batch, c_len, q_len)

        c_masks = c_masks.view(batch_size, c_len, 1)
        q_masks = q_masks.view(batch_size, 1, q_len)

        s1 = masked_softmax(s, q_masks, dim=2) # shape (batch, c_len, q_len)
        s2 = masked_softmax(s, c_masks, dim=1) # shape (batch, c_len, q_len)

        a = torch.bmm(s1, question) # shape (batch, c_len, hidden_size)

        ss = torch.bmm(s1, s2.transpose(1, 2)) # shape (batch, c_len, c_len)
        b = torch.bmm(ss, context) # shape (batch, c_len, hidden_size)

        x = torch.cat([context, a, context * a, context * b], dim=2)

        return x

    def get_similarity_matrix(self, context, question):
        c_len = context.shape[1]
        q_len = question.shape[1]

        s0 = torch.matmul(context, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(question, self.q_weight).transpose(1, 2).expand([-1, c_len, -1])
        s2 = torch.matmul(context * self.cq_weight, question.transpose(1, 2))

        s = s0 + s1 + s2 + self.b # shape (batch, c_len, q_len)

        return s


class Output(nn.Module):
    """
    Output layer
    """

    def __init__(self, hidden_size, drop_prob=0.0):
        super(Output, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = Encoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
