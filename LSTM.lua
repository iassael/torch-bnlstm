--[[

    Recurrent Batch Normalization
    Tim Cooijmans, Nicolas Ballas, César Laurent, Çağlar Gülçehre, Aaron Courville
    http://arxiv.org/abs/1603.09025

    Implemented by Yannis M. Assael (www.yannisassael.com), 2016.

    Based on https://github.com/wojciechz/learning_to_execute 
    and Brendan Shillingford.

    Usage: 
    local rnn = nn.LSTM(rnn_size, input_size, use_bn)
    
]]--

require 'nn'
require 'nngraph'
require 'LinearNB'

function nn.LSTM(rnn_size, input_size, use_bn)
    if not input_size then
        input_size = rnn_size
    end

    local wx_bn, wh_bn, c_bn
    if use_bn then
        wx_bn = nn.BatchNormalization(4 * rnn_size)
        wh_bn = nn.BatchNormalization(4 * rnn_size)
        c_bn = nn.BatchNormalization(rnn_size)
    else
        wx_bn = nn.Identity()
        wh_bn = nn.Identity()
        c_bn = nn.Identity()
    end

    local x = nn.Identity()()
    local prev_state = nn.Identity()()
    local prev_c, prev_h = prev_state:split(2)

    local x_all = nn.View(rnn_size, 4):setNumInputDims(1)(wx_bn(nn.LinearNB(input_size, 4 * rnn_size)(x)))

    local h_all = nn.View(rnn_size, 4):setNumInputDims(1)(wh_bn(nn.Linear(rnn_size, 4 * rnn_size)(prev_h)))

    local sum_all = nn.CAddTable() { x_all, h_all }
    local sum_splits = { nn.SplitTable(2, 2)(sum_all):split(4) }
    local in_gate = nn.Sigmoid()(sum_splits[1])
    local forget_gate = nn.Sigmoid()(sum_splits[2])
    local out_gate = nn.Sigmoid()(sum_splits[3])
    local in_transform = nn.Tanh()(sum_splits[4])

    local next_c = nn.CAddTable()({
        nn.CMulTable()({ forget_gate, prev_c }),
        nn.CMulTable()({ in_gate, in_transform })
    })
    local next_h = nn.CMulTable()({ out_gate, nn.Tanh()(c_bn(next_c)) })
    local next_state = nn.Identity() { next_c, next_h }

    nngraph.annotateNodes()

    return nn.gModule({ x, prev_state }, { next_h, next_state })
end

