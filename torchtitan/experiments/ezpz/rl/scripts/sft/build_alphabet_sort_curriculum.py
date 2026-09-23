#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build a deterministic, pre-tokenized AGPT alphabet-sort SFT curriculum."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

START_USER = "<start_of_turn>user\n"
END_TURN = "<end_of_turn>\n"
START_MODEL = "<start_of_turn>model\n"
OPEN_TAG = "<alphabetical_sorted>"
CLOSE_TAG = "</alphabetical_sorted>"

# Evaluation names and the formatting-example names are deliberately absent.
FIRST_NAMES = (
    "Ada Amir Ana Ava Ben Cara Chen Dalia Diego Elena Emma Farah Felix Grace Hana "
    "Hugo Iris Ivan Jade Jamal Jonah Julia Kai Kira Lars Leah Leo Lina Luca Maya "
    "Mina Nadia Noah Nora Owen Priya Ravi Rosa Ruby Samira Theo Uma Vera Yara Zane"
).split()
LAST_NAMES = (
    "Adams Ahmed Baker Bell Carter Chen Costa Diaz Evans Ford Garcia Gupta Hall Ito "
    "Jones Khan Kim Lee Lopez Martin Moore Novak Patel Reed Rossi Singh Smith Stone "
    "Tanaka Torres Wang White Wilson Wu Yamamoto Zhao"
).split()
INSTRUCTIONS = (
    "Sort these names in alphabetical order by FIRST name: {names}\n\n"
    "Reply with ONLY this block, one name per line, nothing before or after:\n"
    f"{OPEN_TAG}\nName1\nName2\n...\n{CLOSE_TAG}",
    "Alphabetize the following names by first name: {names}\n\n"
    "Return only the completed block:\n"
    f"{OPEN_TAG}\nName1\nName2\n...\n{CLOSE_TAG}",
    "Put these people in ascending alphabetical order using their first names: "
    "{names}\n\nOutput exactly:\n"
    f"{OPEN_TAG}\nName1\nName2\n...\n{CLOSE_TAG}",
)
DISTRACTOR_INSTRUCTIONS = (
    "\n\nExample of the required output format only; its names are unrelated and "
    "must not appear in your answer:\n",
    "\n\nUse the following block only as a formatting illustration. Ignore every "
    "name in it and answer for the requested list:\n",
)
FORBIDDEN = {
    "BethMillar",
    "ShahramKhosravi",
    "AliceZimmer",
    "BobYoung",
    "QuinnRivera",
    "OmarSaito",
    "PiaValdez",
}


def _make_names(rng: random.Random, count: int) -> list[str]:
    candidates = [f"{first}{last}" for first in FIRST_NAMES for last in LAST_NAMES]
    candidates = [name for name in candidates if name not in FORBIDDEN]
    names = rng.sample(candidates, count)
    rng.shuffle(names)
    return names


def build(model: Path, output: Path, size: int, seed: int, robust: bool) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    rng = random.Random(seed)
    rows = {"input_ids": [], "assistant_masks": [], "seq_lengths": []}
    for index in range(size):
        group = index // 8
        count = 1 + (group % 8) if robust else 1 + (index % 8)
        distractor_count = 2 + ((index // 64) % 4)
        sampled = _make_names(
            rng, count + (distractor_count if robust and index % 8 < 2 else 0)
        )
        names = sampled[:count]
        prompt = INSTRUCTIONS[
            group % len(INSTRUCTIONS) if robust else index % len(INSTRUCTIONS)
        ].format(names=", ".join(names))
        if robust and index % 8 < 2:
            distractors = sampled[count:]
            rng.shuffle(distractors)
            prompt += (
                DISTRACTOR_INSTRUCTIONS[index % 2]
                + f"{OPEN_TAG}\n"
                + "\n".join(distractors)
                + f"\n{CLOSE_TAG}"
            )
        ordered = sorted(names, key=lambda name: name.casefold())
        response = f"{OPEN_TAG}\n" + "\n".join(ordered) + f"\n{CLOSE_TAG}"
        prefix = f"{START_USER}{prompt}{END_TURN}{START_MODEL}"
        completion = f"{response}{END_TURN}"
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)
        input_ids = prefix_ids + completion_ids
        if input_ids[0] != 106 or input_ids[-2:] != [107, 108]:
            raise ValueError(
                f"serialization mismatch at row {index}: "
                f"first={input_ids[0]} tail={input_ids[-4:]}"
            )
        rows["input_ids"].append(input_ids)
        rows["assistant_masks"].append(
            [0] * len(prefix_ids) + [1] * len(completion_ids)
        )
        rows["seq_lengths"].append([len(input_ids)])

    dataset = Dataset.from_dict(rows)
    if output.exists():
        shutil.rmtree(output)
    dataset.save_to_disk(output)
    print(
        f"ALPHABET_CURRICULUM_DONE output={output} rows={len(dataset)} "
        f"seed={seed} robust={robust} "
        f"max_length={max(map(len, rows['input_ids']))}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=154391)
    parser.add_argument("--robust", action="store_true")
    args = parser.parse_args()
    build(args.model, args.output, args.size, args.seed, args.robust)


if __name__ == "__main__":
    main()
