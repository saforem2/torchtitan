# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run lm-eval with narrow Transformers 5 compatibility aliases."""

import runpy

import transformers


if not hasattr(transformers, "AutoModelForVision2Seq"):
    transformers._objects[
        "AutoModelForVision2Seq"
    ] = transformers.AutoModelForImageTextToText  # type: ignore[attr-defined]

runpy.run_module("lm_eval", run_name="__main__")
