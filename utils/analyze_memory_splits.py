#! /usr/bin/env python
"""
Analyze the data splits for memory dialogs.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json


def main(args):
    splits = ["train", "val", "test"]
    memory_data = {}
    for split in splits:
        with open(args[f"input_{split}_json"], "r") as file_id:
            memory_data[split] = json.load(file_id)

    for split, data in memory_data.items():
        num_dialogs = len(data["dialogue_data"])
        num_utterances = sum(len(ii["dialogue"]) for ii in data["dialogue_data"])
        print(f"{split}:")
        print(f"\t# dialogs {num_dialogs}\n\t# utterances: {num_utterances}")

    splits = ["train", "val", "test"]
    gpt_data = {}
    for split in splits:
        with open(args[f"gpt_{split}_json"], "r") as file_id:
            gpt_data[split] = json.load(file_id)

    for split, data in gpt_data.items():
        num_dialogs = len(set(ii["dialog_id"] for ii in data))
        num_utterances = len(data) // 2
        print(f"{split}:")
        print(f"\t# dialogs {num_dialogs}\n\t# utterances: {num_utterances}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_train_json",
        required=True,
        help="JSON Input train",
    )
    parser.add_argument(
        "--input_val_json",
        required=True,
        help="JSON Input val",
    )
    # parser.add_argument(
    #     "--input_devtest_json",
    #     required=True,
    #     help="JSON Input devtest",
    # )
    parser.add_argument(
        "--input_test_json",
        required=True,
        help="JSON Input test",
    )
    parser.add_argument(
        "--gpt_train_json",
        required=True,
        help="JSON Input train",
    )
    parser.add_argument(
        "--gpt_val_json",
        required=True,
        help="JSON Input val",
    )
    # parser.add_argument(
    #     "--gpt_devtest_json",
    #     required=True,
    #     help="JSON Input devtest",
    # )
    parser.add_argument(
        "--gpt_test_json", required=True, help="JSON Input test",
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
