#!/usr/bin/env python

# Author: Makenzie Brian
# Date: May 27, 2018
# Class: ME 599
# File: encode_decode.py
# Description: encoder and decoder values for deep learning layer calculator into pytorch

import torch
from torch import nn


class AutoEncoderDecoder:
    def __init__(self, parameter_list, encode_sizes, decode_sizes, decode_starter_size):
        self.encoder = encode_sequence(parameter_list, encode_sizes)
        self.decoder = decode_sequence(parameter_list, decode_sizes, decode_starter_size)


def encode_sequence(parameter_list, encode_sizes):
    """inputs values for encoding layers into torch model
    
    """
    encoder = nn.Sequential(
        nn.Conv1d(int(parameter_list[2][1]), encode_sizes[0], int(parameter_list[2][4]), stride=int(parameter_list[2][5]), padding=int(parameter_list[2][2]), dilation=int(parameter_list[2][3])),
        nn.MaxPool1d(int(parameter_list[3][4]), int(parameter_list[3][5])),
        nn.Conv1d(encode_sizes[1], encode_sizes[2], int(parameter_list[4][4]), stride=int(parameter_list[4][5]), padding=int(parameter_list[4][2]), dilation=int(parameter_list[4][3])),
        nn.MaxPool1d(int(parameter_list[5][4]), int(parameter_list[5][5])),
        nn.Conv1d(encode_sizes[3], encode_sizes[4], int(parameter_list[6][4]), stride=int(parameter_list[6][5]), padding=int(parameter_list[6][2]), dilation=int(parameter_list[6][3])),
        nn.MaxPool1d(int(parameter_list[7][4]), int(parameter_list[7][5]))
    )
    
    return encoder


def decode_sequence(parameter_list, decode_sizes, decode_starter_size):
    """inputs values for decoding layers into torch model
    
    """
    decoder = nn.Sequential(
        nn.ConvTranspose1d(decode_starter_size, decode_sizes[0], int(parameter_list[9][4]), stride=int(parameter_list[9][5]), padding=int(parameter_list[9][2]), output_padding=int(parameter_list[9][6]), dilation=int(parameter_list[9][3])), 
        nn.ConvTranspose1d(decode_sizes[0], decode_sizes[1], int(parameter_list[10][4]), stride=int(parameter_list[10][5]), padding=int(parameter_list[10][2]), output_padding=int(parameter_list[10][6]), dilation=int(parameter_list[10][3])), 
        nn.ConvTranspose1d(decode_sizes[1], decode_sizes[2], int(parameter_list[11][4]), stride=int(parameter_list[11][5]), padding=int(parameter_list[11][2]), output_padding=int(parameter_list[11][6]), dilation=int(parameter_list[11][3])), 
        nn.ConvTranspose1d(decode_sizes[2], decode_sizes[3], int(parameter_list[12][4]), stride=int(parameter_list[12][5]), padding=int(parameter_list[12][2]), output_padding=int(parameter_list[12][6]), dilation=int(parameter_list[12][3])), 
        nn.MaxUnpool1d(int(parameter_list[13][4]), int(parameter_list[13][5]), int(parameter_list[13][2])),
        nn.ConvTranspose1d(decode_sizes[4], decode_sizes[5], int(parameter_list[14][4]), stride=int(parameter_list[14][5]), padding=int(parameter_list[14][2]), output_padding=int(parameter_list[14][6]), dilation=int(parameter_list[14][3])) 
    )
    
    return decoder

def save_model(parameter_list, encode_sizes, decode_sizes, decode_starter_size):
    model = AutoEncoderDecoder(parameter_list, encode_sizes, decode_sizes, decode_starter_size).cuda()
    print "File will be saved in current directory\n"
    model_file = raw_input("Please enter a file name (without extension) to save the model to: ")
    model_file = "./" + str(model_file) + ".pth"
    torch.save(model.state_dict(), model_file)
    print('saving model to file {}'.format(model_file))
    

#do something like this
#torch.save(model.state_dict(), params_file)
#print('saved model to file {}'.format(params_file))
#print('We would recommend you rename this model at asap to avoid overwriting.')
