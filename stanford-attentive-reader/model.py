import torch.nn as nn

class Encoder(nn.Module):
    """
    Encoding layer used by Stanford Attentive Reader
    """
    def __init__(self, embeddings, hidden_size):
        super(Encoder, self).__init__()
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True)

    def forward(self, sents):
        """
        Forward pass for the Encoder

        @param sents (List[[List[str]]): batch of input sequences
        """
