#!/usr/bin/env python

# Author: Makenzie Brian
# Date: May 27, 2018
# Class: ME 599
# File: encode_decode_test.py
# Description: tests functions for encoder and decoder values for deep learning layer calculator
# into pytorch


import pytest
from ..encode_decode import *


def test_encode_sequence():
    """ Check that output sequence (as string) is correct based on known input
        there are no edge cases"""
    
    encoder_test_output = encode_sequence([['', 'Input size', 'Padding', 'Dilation', 'Kernel Size', 'Stride', 'Output Padding (decoder only)', 'Output Size'], ['Encoder', '', '', '', '', '', '', ''], ['Conv1d', '360', '1', '1', '5', '3', '', '120'], ['Maxpool1d', '120', '0', '1', '1', '1', '', '120'], ['Conv1d', '120', '1', '1', '5', '3', '', '40'], ['Maxpool1d', '40', '0', '1', '1', '1', '', '40'], ['Conv1d', '40', '0', '1', '5', '5', '', '8'], ['Maxpool1d', '8', '0', '1', '1', '1', '', '8'], ['Decoder ', '', '', '', '', '', '', ''], ['ConvTrans1d', '8', '0', '0', '5', '5', '0', '40'], ['ConvTrans1d', '40', '1', '1', '5', '3', '0', '120'], ['ConvTrans1d', '120', '1', '1', '5', '3', '0', '360']], [120, 120, 40, 40, 8, 8])
    
    assert str(encoder_test_output) == "Sequential(\n  (0): Conv1d(360, 120, kernel_size=(5,), stride=(3,), padding=(1,))\n  (1): MaxPool1d(kernel_size=1, stride=1, padding=0, dilation=1, ceil_mode=False)\n  (2): Conv1d(120, 40, kernel_size=(5,), stride=(3,), padding=(1,))\n  (3): MaxPool1d(kernel_size=1, stride=1, padding=0, dilation=1, ceil_mode=False)\n  (4): Conv1d(40, 8, kernel_size=(5,), stride=(5,))\n  (5): MaxPool1d(kernel_size=1, stride=1, padding=0, dilation=1, ceil_mode=False)\n)"

    

def test_decode_sequence():
    """ Check that output sequence (as string) is correct based on known input
        there are no edge cases"""
       
    decoder_test_output = decode_sequence([['', 'Input size', 'Padding', 'Dilation', 'Kernel Size', 'Stride', 'Output Padding (decoder only)', 'Output Size'], ['Encoder', '', '', '', '', '', '', ''], ['Conv1d', '360', '1', '1', '5', '3', '', '120'], ['Maxpool1d', '120', '0', '1', '1', '1', '', '120'], ['Conv1d', '120', '1', '1', '5', '3', '', '40'], ['Maxpool1d', '40', '0', '1', '1', '1', '', '40'], ['Conv1d', '40', '0', '1', '5', '5', '', '8'], ['Maxpool1d', '8', '0', '1', '1', '1', '', '8'], ['Decoder ', '', '', '', '', '', '', ''], ['ConvTrans1d', '8', '0', '0', '5', '5', '0', '40'], ['ConvTrans1d', '40', '1', '1', '5', '3', '0', '120'], ['ConvTrans1d', '120', '1', '1', '5', '3', '0', '360']], [40, 120, 360], 8)
    
    assert str(decoder_test_output) == "Sequential(\n  (0): ConvTranspose1d(8, 40, kernel_size=(5,), stride=(5,), dilation=(0,))\n  (1): ConvTranspose1d(40, 120, kernel_size=(5,), stride=(3,), padding=(1,))\n  (2): ConvTranspose1d(120, 360, kernel_size=(5,), stride=(3,), padding=(1,))\n)"
    
    
#def test_save_model(capsys):
    """ CURRENTLY BROKEN BECAUSE OUTPUT WILL NOT GET CAPTURED even though this should work
        Only tests that function will run correctly
        could not get further tests working with Travis to check is save was working correctly
        but did test manually"""
#    with capsys.disabled():
#        save_model([['', 'Input size', 'Padding', 'Dilation', 'Kernel Size', 'Stride', 'Output Padding (decoder only)', 'Output Size'], ['Encoder', '', '', '', '', '', '', ''], ['Conv1d', '360', '1', '1', '5', '3', '', '120'], ['Maxpool1d', '120', '0', '1', '1', '1', '', '120'], ['Conv1d', '120', '1', '1', '5', '3', '', '40'], ['Maxpool1d', '40', '0', '1', '1', '1', '', '40'], ['Conv1d', '40', '0', '1', '5', '5', '', '8'], ['Maxpool1d', '8', '0', '1', '1', '1', '', '8'], ['Decoder ', '', '', '', '', '', '', ''], ['ConvTrans1d', '8', '0', '0', '5', '5', '0', '40'], ['ConvTrans1d', '40', '1', '1', '5', '3', '0', '120'], ['ConvTrans1d', '120', '1', '1', '5', '3', '0', '360']], [120, 120, 40, 40, 8, 8], [40, 120, 360], 8)


    
