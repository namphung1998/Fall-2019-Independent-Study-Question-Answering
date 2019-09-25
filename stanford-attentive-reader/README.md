# Stanford Attentive Reader

An implementation of the Stanford Attentive Reader model (Chen et al, 2016)
for the reading comprehension task. This model uses an RNN-based encoder-decoder
architecture, along with attention mechanism to determine the specific parts
of the context paragraph to attend to.

## Layers

### 1. Encoder

The encoder is an RNN that takes in the input sequence and produce a representation
for each token in the sequence. In particular, we use a shallow bi-directional
LSTM with hidden size $h$.
