#!/usr/bin/env python

# Author: Makenzie Brian
# Date: May 29, 2018
# Class: ME 599
# File: calculate_sizes.py
# Description: calculations for deep learning layer calculator project

import csv


def import_all(filename):
    """ Imports file contents from user with parameters from nueral net and later calculations
        currently not robust to missing or incorrect arguments from file
        currently does not convert values to int; done in later functions
        inputs: filename - name of input file, currently as inputs.csv
        output: file_list - list of parameters specified by user"""
        
    with open(filename,'r') as file:
        file_read = csv.reader(file)
        file_list = []

        for line in file_read:
            file_list.append(line)

    print "Imported file: " + str(filename)
    return file_list
    
    
def calculate_output_sizes_encoder(param_list):
    """ Calculates output sizes for each encoding layer based on user given parameters
        inputs: param_list - values from file given by user
        output: encoder_sizes_list - calculated sizes for encoder layers
                encoder_maxpool1d_output_size_layer_six - calculated size for final encoder layer"""
    
    encoder_conv1d_output_size_layer_one = 1 + ((int(param_list[2][1]) + (2 * int(param_list[2][2])) - (int(param_list[2][3]) * (int(param_list[2][4]) - 1)) - 1) / (int(param_list[2][5])))
    encoder_maxpool1d_output_size_layer_two = 1 + ((encoder_conv1d_output_size_layer_one + (2 * int(param_list[3][2])) - (int(param_list[3][3]) * (int(param_list[3][4]) - 1)) - 1) / (int(param_list[3][5])))
    encoder_conv1d_output_size_layer_three = 1 + ((encoder_maxpool1d_output_size_layer_two + (2 * int(param_list[4][2])) - (int(param_list[4][3]) * (int(param_list[4][4]) - 1)) - 1) / (int(param_list[4][5])))
    encoder_maxpool1d_output_size_layer_four = 1 + ((encoder_conv1d_output_size_layer_three + (2 * int(param_list[5][2])) - (int(param_list[5][3]) * (int(param_list[5][4]) - 1)) - 1) / (int(param_list[5][5])))
    encoder_conv1d_output_size_layer_five = 1 + ((encoder_maxpool1d_output_size_layer_four + (2 * int(param_list[6][2])) - (int(param_list[6][3]) * (int(param_list[6][4]) - 1)) - 1) / (int(param_list[6][5])))
    encoder_maxpool1d_output_size_layer_six = 1 + ((encoder_conv1d_output_size_layer_five + (2 * int(param_list[7][2])) - (int(param_list[7][3]) * (int(param_list[7][4]) - 1)) - 1) / (int(param_list[7][5])))
    
    encoder_sizes_list =[encoder_conv1d_output_size_layer_one, encoder_maxpool1d_output_size_layer_two, encoder_conv1d_output_size_layer_three, encoder_maxpool1d_output_size_layer_four, encoder_conv1d_output_size_layer_five, encoder_maxpool1d_output_size_layer_six]
    
    return encoder_sizes_list, encoder_maxpool1d_output_size_layer_six
    
    
def calculate_output_sizes_decoder(param_list, encode_final_layer_size):
    """ Calculates output sizes for each decoding layer based on user given parameters
        inputs: param_list - values from file given by user
                encode_final_layer_size - calculated size for final encoder layer
        output: decoder_sizes_list - calculated sizes for decoder layers"""
    
    decoder_convtrans1d_output_size_layer_one = ((encode_final_layer_size - 1) * int(param_list[9][5])) - (2 * int(param_list[9][2])) + int(param_list[9][4]) + int(param_list[9][6])
    decoder_convtrans1d_output_size_layer_two = ((decoder_convtrans1d_output_size_layer_one - 1) * int(param_list[10][5])) - (2 * int(param_list[10][2])) + int(param_list[10][4]) + int(param_list[10][6])
    decoder_convtrans1d_output_size_layer_three = ((decoder_convtrans1d_output_size_layer_two - 1) * int(param_list[11][5])) - (2 * int(param_list[11][2])) + int(param_list[11][4]) + int(param_list[11][6])
    
    decoder_sizes_list = [decoder_convtrans1d_output_size_layer_one, decoder_convtrans1d_output_size_layer_two, decoder_convtrans1d_output_size_layer_three]
    
    return decoder_sizes_list
    
    
    
    
    
