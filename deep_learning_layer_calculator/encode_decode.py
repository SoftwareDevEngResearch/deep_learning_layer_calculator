#!/usr/bin/env python

# Author: Makenzie Brian
# Date: May 27, 2018
# Class: ME 599
# File: encode_decode.py
# Description: encoder and decoder values for deep learning layer calculator into pytorch nueral
# net model

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

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


def plot_model(parameter_list, encode_sizes, decode_sizes):
    """ """
    fig, ax = plt.subplots(1)
    ax.set_xlim(0, 280)
    ax.set_ylim(-((int(parameter_list[2][1])/2)+430), (int(parameter_list[2][1])/2)+30)
    plt.axis('off')
    
    rect_encode_layer_one = Rectangle((0, -int(parameter_list[2][1])/2), 20, int(parameter_list[2][1]))
    rect_encode_layer_two = Rectangle((30, -encode_sizes[0]/2), 20, encode_sizes[0])
    rect_encode_layer_three = Rectangle((60, -encode_sizes[1]/2), 20, encode_sizes[1])
    rect_encode_layer_four = Rectangle((90, -encode_sizes[2]/2), 20, encode_sizes[2])
    rect_encode_layer_five = Rectangle((120, -encode_sizes[3]/2), 20, encode_sizes[3])
    rect_encode_layer_six = Rectangle((150, -encode_sizes[4]/2), 20, encode_sizes[4])
    
    rect_decode_layer_one = Rectangle((180, -decode_sizes[0]/2), 20, decode_sizes[0])
    rect_decode_layer_two = Rectangle((210, -decode_sizes[1]/2), 20, decode_sizes[1])
    rect_decode_layer_three = Rectangle((240, -decode_sizes[2]/2), 20, decode_sizes[2])
    
    ax.text(7, -(100 + int(parameter_list[2][1])/2), 'Conv1d  First Layer', rotation='vertical')
    ax.text(37, -(100 + (encode_sizes[0]/2)), 'MaxPool1d  Second Layer', rotation='vertical')
    ax.text(67, -(100 + (encode_sizes[1]/2)), 'Conv1d  Third Layer', rotation='vertical')
    ax.text(97, -(100 + (encode_sizes[2]/2)), 'MaxPool1d  Fourth Layer', rotation='vertical')
    ax.text(127, -(100 + (encode_sizes[3]/2)), 'Conv1d  Fifth Layer', rotation='vertical')
    ax.text(157, -(100 + (encode_sizes[4]/2)), 'MaxPool1d  Sixth Layer', rotation='vertical')
    
    ax.text(187, -(100 + (decode_sizes[0]/2)), 'ConvTranspose1d  First Layer', rotation='vertical')
    ax.text(217, -(100 + (decode_sizes[1]/2)), 'ConvTranspose1d  Second Layer', rotation='vertical')
    ax.text(247, -(100 + (decode_sizes[2]/2)), 'ConvTranspose1d  Third Layer', rotation='vertical')
    
    ax.text(53, (int(parameter_list[2][1])/2)+20, 'Encoder Layers')
    
    ax.text(187, (int(parameter_list[2][1])/2)+20, 'Decoder Layers')
    
    rectangles = [rect_encode_layer_one, rect_encode_layer_two, rect_encode_layer_three, rect_encode_layer_four, rect_encode_layer_five, rect_encode_layer_six, rect_decode_layer_one, rect_decode_layer_two, rect_decode_layer_three]
    pc = PatchCollection(rectangles)
    ax.add_collection(pc)
    
    plt.show()
    

