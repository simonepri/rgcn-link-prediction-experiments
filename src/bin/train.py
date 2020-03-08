#!/usr/bin/python3
import argparse
import os
import random
import sys
import unittest

import numpy as np
import tensorflow as tf

from mocks.model_builder import build_decoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model on a given dataset."
    )
    parser.add_argument(
        "--settings",
        type=str,
        required=True,
        help="Filepath for settings file.",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Filepath for dataset."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


def run_in_submodule(function):
    # HACK: The submodule needs these path to be run
    sys.path.insert(0, "src/submodules/rgcn/code/optimization")
    sys.path.insert(0, "src/submodules/rgcn/code")
    function()
    sys.path.remove("src/submodules/rgcn/code/optimization")
    sys.path.remove("src/submodules/rgcn/code")


def run_mocked_train():
    @unittest.mock.patch("common.model_builder.build_decoder", build_decoder)
    def run(args):
        # Disable most of the TF warnings
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        # Set random seed for riproducibility
        set_seed(args.seed)
        # Set arguments
        sys.argv = [
            "src/submodules/rgcn/code/train.py",
            "--settings",
            args.settings,
            "--dataset",
            args.dataset,
        ]
        # pylint: disable=import-outside-toplevel,unused-import
        import train

    ARGS = parse_args()
    run(ARGS)


run_in_submodule(run_mocked_train)
