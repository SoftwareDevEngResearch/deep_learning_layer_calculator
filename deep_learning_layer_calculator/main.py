#!/usr/bin/env python

# Author: Makenzie Brian
# Date: May 25, 2018
# Class: ME 599
# File: main.py
# Description: main file for deep leanring calculator project

from encode_decode import *
from calculate_sizes import *

if __name__ == "__main__":
    parameter_list = import_all('inputs.csv')
    encode_list, decode_starter_size = calculate_output_sizes_encoder(parameter_list)
    decode_list = calculate_output_sizes_decoder(parameter_list, decode_starter_size)
    encode_sequence(parameter_list, encode_list)
    decode_sequence(parameter_list, decode_list, decode_starter_size)
    
    #print encoder_list, decoder_list
