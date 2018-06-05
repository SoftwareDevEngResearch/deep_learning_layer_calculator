#!/usr/bin/env python

# Author: Makenzie Brian
# Date: May 29, 2018
# Class: ME 599
# File: calculate_sizes.py
# Description: tests for calculations for deep learning layer calculator project

import pytest
from ..calculate_sizes import *


def test_import_all():
    """checks if import function is importing things from csv file; not robust
       does not check if ourput is correct because could not get file to be found in travis"""
       
    assert import_all('inputs.csv') != None
    
def test_calculate_output_sizes_encoder():
    """compares calculated values to known set; all formulas are the same so includes variety
       no real edge cases but does test with some zeros"""
       
    encoder_test_output, last_layer_test_output = calculate_output_sizes_encoder([['', 'Input size', 'Padding', 'Dilation', 'Kernel Size', 'Stride', 'Output Padding (decoder only)', 'Output Size'], ['Encoder', '', '', '', '', '', '', ''], ['Conv1d', '360', '1', '1', '5', '3', '', '120'], ['Maxpool1d', '120', '0', '1', '1', '1', '', '120'], ['Conv1d', '120', '1', '1', '5', '3', '', '40'], ['Maxpool1d', '40', '0', '1', '1', '1', '', '40'], ['Conv1d', '40', '0', '1', '5', '5', '', '8'], ['Maxpool1d', '8', '0', '1', '1', '1', '', '8'], ['Decoder ', '', '', '', '', '', '', ''], ['ConvTrans1d', '8', '0', '0', '5', '5', '0', '40'], ['ConvTrans1d', '40', '1', '1', '5', '3', '0', '120'], ['ConvTrans1d', '120', '1', '1', '5', '3', '0', '360']])
    assert encoder_test_output == [120, 120, 40, 40, 8, 8] and last_layer_test_output == 8

def test_calculate_output_sizes_decoder():
    """compares calculated values to known set; all formulas are the same so includes variety
       no real edge cases but does test with some zeros"""
       
    decoder_test_output = calculate_output_sizes_decoder([['', 'Input size', 'Padding', 'Dilation', 'Kernel Size', 'Stride', 'Output Padding (decoder only)', 'Output Size'], ['Encoder', '', '', '', '', '', '', ''], ['Conv1d', '360', '1', '1', '5', '3', '', '120'], ['Maxpool1d', '120', '0', '1', '1', '1', '', '120'], ['Conv1d', '120', '1', '1', '5', '3', '', '40'], ['Maxpool1d', '40', '0', '1', '1', '1', '', '40'], ['Conv1d', '40', '0', '1', '5', '5', '', '8'], ['Maxpool1d', '8', '0', '1', '1', '1', '', '8'], ['Decoder ', '', '', '', '', '', '', ''], ['ConvTrans1d', '8', '0', '0', '5', '5', '0', '40'], ['ConvTrans1d', '40', '1', '1', '5', '3', '0', '120'], ['ConvTrans1d', '120', '1', '1', '5', '3', '0', '360']], 8)
    assert decoder_test_output == [40, 120, 360]


    
