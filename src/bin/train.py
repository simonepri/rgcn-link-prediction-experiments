#!/usr/bin/python3
import os
import sys
import unittest

from mocks.model_builder import build_decoder


def run_in_submodule(function):
    # HACK: The submodule needs these path to be run
    sys.path.insert(0, "src/submodules/rgcn/code/optimization")
    sys.path.insert(0, "src/submodules/rgcn/code")
    function()
    sys.path.remove("src/submodules/rgcn/code/optimization")
    sys.path.remove("src/submodules/rgcn/code")


def run_mocked_train():
    @unittest.mock.patch("common.model_builder.build_decoder", build_decoder)
    def run():
        # Disable most of the TF warnings
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        # pylint: disable=import-outside-toplevel,unused-import
        import train

    run()


run_in_submodule(run_mocked_train)
