#!/usr/bin/env python

from collections import OrderedDict
from bitstring import BitArray


def get_parameters(train=None, dev=None, test=None, tag_scheme='iobes', lower=False, zeros=False, char_dim=25,
             char_lstm_dim=25, char_bidirect=True, word_dim=100, word_lstm_dim=100, word_bidirect=True, pre_emb=None,
             all_emb=False, cap_dim=0, crf=True, dropout=0.5, lr_method='sgd-lr_.005', reload=False):
    parameters = OrderedDict()
    parameters['train'] = train
    parameters['dev'] = dev
    parameters['test'] = test
    parameters['tag_scheme'] = tag_scheme
    parameters['lower'] = lower  #1 bit
    parameters['zeros'] = zeros  #1 bit
    parameters['char_dim'] = char_dim
    parameters['char_lstm_dim'] = char_lstm_dim  # 6 bits
    parameters['char_bidirect'] = char_bidirect  # 1 bit
    parameters['word_dim'] = word_dim
    parameters['word_lstm_dim'] = word_lstm_dim  # 8 bits
    parameters['word_bidirect'] = word_bidirect  # 1 bit
    parameters['pre_emb'] = pre_emb
    parameters['all_emb'] = all_emb
    parameters['cap_dim'] = cap_dim  # 1 bit
    parameters['crf'] = crf
    parameters['dropout'] = dropout
    parameters['lr_method'] = lr_method
    parameters['reload'] = reload
    return parameters


def get_parameters_from_individual(ga_individual_solution, train=None, dev=None, test=None, tag_scheme='iobes', char_dim=25, word_dim=100,
        pre_emb=None, all_emb=False, crf=True, dropout=0.5, lr_method='sgd-lr_.005', reload=False):
    # Decode GA solution to integer for lower, zeros, char_lstm_dim, char_bidirect, word_lstm_dim, word_bidirect
    # and cap_dim
    lower_bit = ga_individual_solution[0]
    zeros_bit = ga_individual_solution[1]
    char_lstm_dim_bits = BitArray(ga_individual_solution[2:8])
    char_bidirect_bit = ga_individual_solution[8]
    word_lstm_dim_bits = BitArray(ga_individual_solution[9:17])
    word_bidirect_bit = ga_individual_solution[17]
    cap_dim_bit = ga_individual_solution[18]
    lower = lower_bit == 1
    zeros = zeros_bit == 1
    char_lstm_dim = char_lstm_dim_bits.uint
    char_bidirect = char_bidirect_bit == 1
    word_lstm_dim = word_lstm_dim_bits.uint
    word_bidirect = word_bidirect_bit == 1
    cap_dim = cap_dim_bit
    return get_parameters(train=train, dev=dev, test=test, tag_scheme=tag_scheme, lower=lower, zeros=zeros,
                          char_dim=char_dim, char_lstm_dim=char_lstm_dim, char_bidirect=char_bidirect,
                          word_dim=word_dim, word_lstm_dim=word_lstm_dim, word_bidirect=word_bidirect, pre_emb=pre_emb,
                          all_emb=all_emb, cap_dim=cap_dim, crf=crf, dropout=dropout, lr_method=lr_method,
                          reload=reload)
