# Recurrent Batch Normalization
Batch-Normalized LSTMs

Tim Cooijmans, Nicolas Ballas, César Laurent, Çağlar Gülçehre, Aaron Courville

[http://arxiv.org/abs/1603.09025](http://arxiv.org/abs/1603.09025)

### Usage
`local rnn = nn.LSTM(input_size, rnn_size, n, dropout, bn)`

n = number of layers (1-N)

dropout = probability of dropping a neuron (0-1)

bn = batch normalization (true, false)

### Example
[https://github.com/iassael/char-rnn](https://github.com/iassael/char-rnn)


Implemented in Torch by Yannis M. Assael (www.yannisassael.com)