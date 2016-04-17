# Recurrent Batch Normalization
Batch-Normalized LSTMs

Tim Cooijmans, Nicolas Ballas, César Laurent, Çağlar Gülçehre, Aaron Courville

[http://arxiv.org/abs/1603.09025](http://arxiv.org/abs/1603.09025)

### Usage
`local rnn = LSTM(input_size, rnn_size, n, dropout, bn)`

n = number of layers (1-N)

dropout = probability of dropping a neuron (0-1)

bn = batch normalization (true, false)

### Example
[https://github.com/iassael/char-rnn](https://github.com/iassael/char-rnn)

### Performance
Validation scores on char-rnn with default options

<img src="http://blog.yannisassael.com/wp-content/uploads/2016/04/bnlstm_val_loss-1024x631.png" width=502 height=309 />

Implemented in Torch by Yannis M. Assael (www.yannisassael.com)