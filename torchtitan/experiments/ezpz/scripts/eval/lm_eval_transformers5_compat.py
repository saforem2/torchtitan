# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run lm-eval with narrow Transformers 5 compatibility aliases."""

import runpy

import transformers


def main() -> None:
    if not hasattr(transformers, "AutoModelForVision2Seq"):
        transformers._objects[  # type: ignore[attr-defined]
            "AutoModelForVision2Seq"
        ] = transformers.AutoModelForImageTextToText
    runpy.run_module("lm_eval", run_name="__main__")


if __name__ == "__main__":
    main()
