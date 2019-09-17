import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from Encoder import Encoder
from Attention import Attention

class AttentiveReaderModel(nn.Module):
    def __init__(self, embeddings, hidden_size):
        super(AttentiveReaderModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(*embeddings.shape)
        self.embedding.weight = nn.Parameter(torch.tensor(embeddings))
        self.encoder = Encoder(hidden_size)
        self.attention = Attention(hidden_size * 2)

    def forward(self):
        pass


