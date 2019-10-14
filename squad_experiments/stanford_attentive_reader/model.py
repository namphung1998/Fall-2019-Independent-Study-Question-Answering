import torch
import torch.nn as nn
import stanford_attentive_reader.layers as layers


class StanfordAttentiveReader(nn.Module):
    """
    Stanford Attentive Reader model for question answering
    """
    def __init__(self, embeddings, hidden_size, drop_prob=0.0):
        super(StanfordAttentiveReader, self).__init__()
        self.input_size = embeddings.shape[1]
        self.embedding = layers.Embedding(embeddings)
        self.encoder = layers.Encoder(input_size=self.input_size, hidden_size=hidden_size, num_layers=1, drop_prob=drop_prob)
        self.attn = layers.Attention(hidden_size=2 * hidden_size)
        self.output = layers.Output(hidden_size=2 * hidden_size)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.embedding(cw_idxs)
        q_emb = self.embedding(qw_idxs)

        contexts, _ = self.encoder(c_emb, c_len)
        _, questions = self.encoder(q_emb, q_len)

        att = self.attn(contexts, questions, c_mask)

        out = self.output(att, c_mask)

        return out
