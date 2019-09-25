import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GRUGenerator(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, padding_idx, vocab):
        super(GRUGenerator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embed_size = embed_size
        self.vocab = vocab

        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=padding_idx)
        self.gru = nn.GRU(embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, names):
        X = self.vocab.to_input_tensor(names)
        input_lengths = [len(s) for s in names]

        X = self.embedding(X)
        X = pack_padded_sequence(X, input_lengths, batch_first=True)
        hiddens, last_hidden = self.gru(X)
        hiddens, _ = pad_packed_sequence(hiddens, batch_first=True)
        outputs = self.softmax(self.fc(hiddens))

        return outputs



