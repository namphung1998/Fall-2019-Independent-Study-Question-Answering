import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, embeddings, hidden_size, dropout_rate=0.2):
        """

        @param embeddings: Tensor of size (num_words, embed_size)
        """
        self.embedding = nn.Embedding(*embeddings.shape)
        self.embedding.shape = nn.Parameter(torch.tensor(embeddings))
        self.lstm = nn.LSTM(input_size=embeddings.shape[1], hidden_size=hidden_size, bidirectional=True)

    def forward(self, input_padded, input_lengths):
        X = self.embedding(input_padded)
        X = pack_padded_sequence(X, input_lengths)
        enc_hiddens, (last_hidden, last_cell) = self.lstm(X)
        enc_hiddens = pad_packed_sequence(enc_hiddens, batch_first=True)[0]
