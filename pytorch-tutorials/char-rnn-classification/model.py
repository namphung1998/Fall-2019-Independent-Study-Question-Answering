import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GRUClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.gru = nn.GRU(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, input_padded, input_lengths):
        X = pack_padded_sequence(input_padded, input_lengths)
        _, last_hidden = self.gru(X) # last_hidden of shape (1, batch, hidden_size)
        output = self.output(last_hidden)

        return output.squeeze(0)
