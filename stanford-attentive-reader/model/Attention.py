import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, query_encoder_last, context_encoder_hiddens):
        """
        @param query_encoder_last (Tensor): tensor of shape (b, hidden_size)
        @param context_encoder_hiddens (Tensor): tensor shape (b, seq_len, hidden_size)

        @returns alpha_t (Tensor): tensor of shape (b, seq_len)
        """

        context_enc_hiddens_proj = self.attn_projection(context_encoder_hiddens)
        energy = torch.bmm(context_enc_hiddens_proj, torch.unsqueeze(query_encoder_last, -1))
        energy = torch.unsqueeze(energy, -1)
        alpha_t = F.softmax(energy, dim=-1)

        return alpha_t
