# Stanford Attentive Reader

An implementation of the Stanford Attentive Reader model (Chen et al, 2016)
for the reading comprehension task. This model uses an RNN-based encoder-decoder
architecture, along with attention mechanism to determine the specific parts
of the context paragraph to attend to.

## Layers

### 1. Embedding

This layer uses the pretrained GloVe 300-dimensional word embeddings (Pennington et al, 2014). Given
a list of indices $w_1, \ldots, w_m$, the embedding layer uses a lookup table to
convert each index into a word embedding, i.e. $v_1, \ldots, v_m \in \mathbb{R}^{300}$. 

### 2. Encoder

The encoder is an RNN that takes in the input sequence and produce a representation
for each token in the sequence. In particular, we use a shallow bi-directional
LSTM with hidden size $h$. The output of the encoder is the hidden state at
each position, as well as the concatenation of the hidden states at the last
timestep in the forward RNN, and the first timestep in the backward RNN.

$$
\begin{align*}
\overrightarrow{h_{t}} = LSTM(v_t, h_{t-1})
\end{align*}
$$
