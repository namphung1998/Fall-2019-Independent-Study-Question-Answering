"""
Layers for use in Stanford Attentive Reader

Author: Nam Phung
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


class Embedding(nn.Module):
    """
    GloVe embedding layer.
    TODO: With suggestions from the BiDAF paper, this word-level
    embedding layer is augmented with a 2-layer Highway Network
    """

    def __init__(self, embeddings):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings)

    def forward(self, x):
        """

        :param x: (Tensor) tensor of shape (batch_size, seq_len)
        :returns embedded: (Tensor) tensor of shape (batch_size, seq_len, embed_size)
        """
        embedded = self.embedding(x)
        return embedded


class Encoder(nn.Module):
    """
    A bi-LSTM network for encoding a sequence. The output of this layer is
    the LSTM's hidden state at each position, which is a tensor of shape
    (batch_size, seq_len, hidden_size * 2)
    """

    def __init__(self, input_size, hidden_size, num_layers, drop_prob=0.0):
        """
        :param input_size: (int) Size of a timestep in the input
        :param hidden_size: (int) Size of RNN hidden state
        :param num_layers: (int) Number of layers
        :param drop_prob: (float) Dropout probability
        """
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True,
                            dropout=drop_prob if num_layers > 1 else 0.0)

    def forward(self, x, lengths):
        """
        Single forward pass of the encoder network
        :param x: (Tensor) input tensor of shape (batch_size, seq_len, input_size)
        :param lengths: (Tensor) the lengths of all sequences in the current batch
        :returns: enc_hiddens (Tensor) tensor of shape (batch_size, seq_len, hidden_size * 2)
        :returns: last_hidden (Tensor) tensor of shape (batch_size, hidden_size * 2)
        """

        orig_len = x.shape[1]

        lengths, sort_idx = torch.sort(lengths, dim=0, descending=True)
        x = x[sort_idx]
        x = pack_padded_sequence(x, lengths, batch_first=True)

        self.lstm.flatten_parameters()
        enc_hiddens, (last_hidden, last_cell) = self.lstm(x) # enc_hiddens is a PackedSequence object
                                                             # last_hidden is of shape (num_layers * num_directions, batch_size, hidden_size)
        enc_hiddens, _ = pad_packed_sequence(enc_hiddens, batch_first=True, total_length=orig_len)
        _, unsort_idx = torch.sort(sort_idx, dim=0)
        enc_hiddens = enc_hiddens[unsort_idx] # enc_hiddens is now a Tensor of shape (batch_size, seq_len, 2 * hidden_size)

        enc_hiddens = F.dropout(enc_hiddens, self.drop_prob, self.training)

        last_hidden = last_hidden.view(self.num_layers, 2, -1, self.hidden_size)[-1, 0] # shape (batch_size, hidden_size)

        last_hidden_reverse = enc_hiddens[:, 0, self.hidden_size:]
        last_hidden = torch.cat([last_hidden, last_hidden_reverse], dim=-1) # shape (batch_size, 2 * hidden_size)

        # last_hidden = torch.reshape(last_hidden, (-1, self.num_layers, self.hidden_size * 2))
        # last_hidden = torch.sum(last_hidden, dim=1)
        # last_hidden = torch.squeeze(last_hidden, 1)

        #TODO: fix last_hidden.shape to be (batch_size, 2 * hidden_size)
        return enc_hiddens, last_hidden


class Attention(nn.Module):
    """
    An attention layer that compares the question embeddings and all the contextual embeddings,
    and selects the pieces of information that are relevant to the question.

    The output of this layer is a tensor of shape (batch_size, seq_len, hidden_size)
    """

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn_proj_1 = nn.Linear(hidden_size, hidden_size)
        self.attn_proj_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, context, question, c_masks):
        """
        Single forward pass of the attention layer
        :param context: (Tensor) tensor of shape (batch_size, context_len, hidden_size),
                                output of encoder layer on context
        :param question: (Tensor) tensor of shape (batch_size, hidden_size), output of encoder layer on context
        :param c_masks:
        :param q_masks:
        :returns: output_t (Tensor) tensor of shape (batch_size, seq_len, hidden_size)
        """

        # context_hidden_proj = self.attn_proj(context) # shape (batch_size, context_len, hidden_size)
        logits_1 = torch.bmm(self.attn_proj_1(context), torch.unsqueeze(question, 2)) # shape (batch_size, context_len, 1)
        logits_1 = torch.squeeze(logits_1, -1)

        logits_2 = torch.bmm(self.attn_proj_2(context), torch.unsqueeze(question, 2)) # shape (batch_size, context_len, 1)
        logits_2 = torch.squeeze(logits_2, -1)

        log_p1 = masked_softmax(logits_1, c_masks, dim=1, log_softmax=True)
        log_p2 = masked_softmax(logits_2, c_masks, dim=1, log_softmax=True)
        #
        # alpha_t = masked_softmax(scores, c_masks, dim=1) # shape (batch_size, context_len)
        # output_t = torch.mul(context, alpha_t.unsqueeze(2))
        #
        # output_t = torch.bmm(torch.unsqueeze(alpha_t, 1), context) # shape (batch_size, 1, hidden_size)
        # output_t = output_t.squeeze(1)
        return log_p1, log_p2


class Output(nn.Module):
    """
    Output layer
    """

    def __init__(self, hidden_size, drop_prob=0.0):
        super(Output, self).__init__()
        self.drop_prob = drop_prob
        self.output_proj1 = nn.Linear(hidden_size, 1)
        self.output_proj2 = nn.Linear(hidden_size, 1)

    def forward(self, att, masks):
        """

        :param att: output of the attention layer, shape (batch_size, seq_len, hidden_size)
        :param masks:
        :return:
        """
        logits1 = F.dropout(self.output_proj1(att), self.drop_prob)
        logits2 = F.dropout(self.output_proj2(att), self.drop_prob)

        log_p1 = masked_softmax(logits1.squeeze(-1), masks, log_softmax=True)
        log_p2 = masked_softmax(logits2.squeeze(-1), masks, log_softmax=True)

        return log_p1, log_p2
