import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.2):
        """

        @param embeddings: Tensor of size (num_words, embed_size)
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=embeddings.shape[1], hidden_size=hidden_size, bidirectional=True)

    def forward(self, X):
        """

        """
        enc_hiddens, last_hidden = self.gru(X)
        enc_hiddens, _ = pad_packed_sequence(enc_hiddens, batch_first=True) # shape (batch, seq_len, input_size)

        return enc_hiddens, last_hidden
