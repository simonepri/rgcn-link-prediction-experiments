#!/usr/bin/python3
import os
import sys
import unittest

# Add required python path
sys.path.insert(0, "src/submodules/rgcn/code")
sys.path.insert(0, "src/submodules/rgcn/code/optimization")

from mocks.model_builder import build_decoder

# Disable most of TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

@unittest.mock.patch('common.model_builder.build_decoder', build_decoder)
def run():
    import submodules.rgcn.code.train

run()
