#!/usr/bin/env python

# Author: Makenzie Brian
# Date: May 27, 2018
# Class: ME 599
# File: encode_decode.py
# Description: encoder and decoder values for deep learning layer calculator into pytorch nueral
# net model

import torch
import torch.nn as nn

class AutoEncoderDecoder(nn.Module):
    """ holds encoder and decoder information for torch model
        contains self function only
        inputs: nn.Module - base class for all neural network modules from torch """
    
    def __init__(self, parameter_list, encode_sizes, decode_sizes, decode_starter_size):
        """ initializes instantiations based on values calculated in following functions
            inputs: parameter_list - values from file given by user
                decode_sizes - calculated sizes for decoder layers
                decode_starter_size - calculated size for final encoder layer"""
                
        super(AutoEncoderDecoder, self).__init__()
        self.encoder = encode_sequence(parameter_list, encode_sizes)
        self.decoder = decode_sequence(parameter_list, decode_sizes, decode_starter_size)


def encode_sequence(parameter_list, encode_sizes):
    """ inputs values for encoding layers into torch neural net model
        inputs: parameter_list - values from file given by user
                encode_sizes - calculated sizes for encoder layers
        output: encoder - encoder structure for neural net"""
    
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
    """ inputs values for decoding layers into torch neural net model
        inputs: parameter_list - values from file given by user
                decode_sizes - calculated sizes for decoder layers
                decode_starter_size - calculated size for final encoder layer
        output: decoder - decoder structure for neural net"""
    
    decoder = nn.Sequential(
        nn.ConvTranspose1d(decode_starter_size, decode_sizes[0], int(parameter_list[9][4]), stride=int(parameter_list[9][5]), padding=int(parameter_list[9][2]), output_padding=int(parameter_list[9][6]), dilation=int(parameter_list[9][3])), 
        nn.ConvTranspose1d(decode_sizes[0], decode_sizes[1], int(parameter_list[10][4]), stride=int(parameter_list[10][5]), padding=int(parameter_list[10][2]), output_padding=int(parameter_list[10][6]), dilation=int(parameter_list[10][3])), 
        nn.ConvTranspose1d(decode_sizes[1], decode_sizes[2], int(parameter_list[11][4]), stride=int(parameter_list[11][5]), padding=int(parameter_list[11][2]), output_padding=int(parameter_list[11][6]), dilation=int(parameter_list[11][3]))        
    )
    
    return decoder


def save_model(parameter_list, encode_sizes, decode_sizes, decode_starter_size):
    """ Saves torch model instantiation to file
        File can be used as structure to train neural net
        Output file name must be chosen by user to avoid overwriting old files
        inputs: parameter_list - values from file given by user
                encode_sizes - calculated sizes for encoder layers
                decode_sizes - calculated sizes for decoder layers
                decode_starter_size - calculated size for final encoder layer
                user input - user must input name of file they want to save structure to
        output: returns 0 to indicate finished
                saves file with specificed name to current folder"""
    
    model = AutoEncoderDecoder(parameter_list, encode_sizes, decode_sizes, decode_starter_size)
    print "\nFile will be saved in current directory"
    model_file = raw_input("Please enter a file name (without extension) to save the model to: ")
    model_file = "./" + str(model_file) + ".pth"
    torch.save(model.state_dict(), model_file)
    print('saving model to file {}'.format(model_file))
    
    return 0
    

