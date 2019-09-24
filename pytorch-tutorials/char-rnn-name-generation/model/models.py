import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GRUGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUGenerator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru = nn.GRU(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_padded, lengths):
        X = pack_padded_sequence(input_padded, lengths)
        hiddens, last_hidden = self.gru(X) 
        hiddens, _ = pad_packed_sequence(hiddens, batch_first=True) # hiddens has shape (batch, seq_len, hidden_size)
        outputs = self.softmax(self.fc(hiddens))

        return outputs