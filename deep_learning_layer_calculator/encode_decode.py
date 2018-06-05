#!/usr/bin/env python

# Author: Makenzie Brian
# Date: May 27, 2018
# Class: ME 599
# File: encode_decode.py
# Description: encoder and decoder values for deep learning layer calculator into pytorch

import torch
from torch import nn

def encode_sequence(parameter_list, encode_sizes):
    """inputs values for encoding layers into torch model
    
    """
    encoder = nn.Sequential(
        nn.Conv1d(int(parameter_list[2][1]), encode_sizes[0], int(parameter_list[2][4]), stride=int(parameter_list[2][5]), padding=int(parameter_list[2][2]), dilation=int(parameter_list[2][3])),
        nn.MaxPool1d(),
        nn.Conv1d(int(parameter_list[4][1]), encode_sizes[2], int(parameter_list[4][4]), stride=int(parameter_list[4][5]), padding=int(parameter_list[4][2]), dilation=int(parameter_list[4][3])),
        nn.MaxPool1d(),
        nn.Conv1d(int(parameter_list[6][1]), encode_sizes[4], int(parameter_list[6][4]), stride=int(parameter_list[6][5]), padding=int(parameter_list[6][2]), dilation=int(parameter_list[6][3])),
        nn.MaxPool1d()
    )
    
    return encoder


def decode_sequence(parameter_list, decode_sizes):
    """inputs values for decoding layers into torch model
    
    """
    decoder = nn.Sequential(
        nn.ConvTranspose1d(int(parameter_list[9][1]), decode_sizes[0], int(parameter_list[9][4]), stride=int(parameter_list[9][5]), padding=int(parameter_list[9][2]), output_padding=int(parameter_list[9][6]), dilation=int(parameter_list[9][3])), 
        nn.ConvTranspose1d(int(parameter_list[10][1]), decode_sizes[1], int(parameter_list[1][4]), stride=int(parameter_list[10][5]), padding=int(parameter_list[10][2]), output_padding=int(parameter_list[10][6]), dilation=int(parameter_list[10][3])), 
        nn.ConvTranspose1d(int(parameter_list[11][1]), decode_sizes[2], int(parameter_list[11][4]), stride=int(parameter_list[11][5]), padding=int(parameter_list[11][2]), output_padding=int(parameter_list[11][6]), dilation=int(parameter_list[11][3])), 
        nn.ConvTranspose1d(int(parameter_list[12][1]), decode_sizes[3], int(parameter_list[12][4]), stride=int(parameter_list[12][5]), padding=int(parameter_list[12][2]), output_padding=int(parameter_list[12][6]), dilation=int(parameter_list[12][3])), 
        nn.MaxUnpool1d(int(parameter_list[13][4]), int(parameter_list[13][5]), int(parameter_list[13][2])),
        nn.ConvTranspose1d(int(parameter_list[14][1]), decode_sizes[5], int(parameter_list[14][4]), stride=int(parameter_list[14][5]), padding=int(parameter_list[14][2]), output_padding=int(parameter_list[14][6]), dilation=int(parameter_list[14][3])) 
    )
    
    return decoder


#do something like this
#torch.save(model.state_dict(), params_file)
#print('saved model to file {}'.format(params_file))
#print('We would recommend you rename this model at asap to avoid overwriting.')
