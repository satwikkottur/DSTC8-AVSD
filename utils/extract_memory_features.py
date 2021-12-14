#! /usr/bin/env python
"""
Extract BUTD features for memories.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import json
import os

import tqdm
import numpy as np


def main(args):
    memory_graphs = {}
    for file_path in args["input_memory_json"]:
        # print(f"Reading: {file_path}")
        with open(file_path, "r") as file_id:
            graph_data = json.load(file_id)
            for datum in graph_data:
                if datum["memory_graph_id"] in memory_graphs:
                    print("Multiple memory graph ids exist!")
                else:
                    memory_graphs[datum["memory_graph_id"]] = datum
    print(f"# memory dialogs: {len(memory_graphs)}")

    memory_dialogs = {}
    for file_path in args["input_dialog_json"]:
        # print(f"Reading: {file_path}")
        with open(file_path, "r") as file_id:
            dialog_data = json.load(file_id)
        for datum in dialog_data["dialogue_data"]:
            dialog_id = datum["dialogue_idx"]
            memory_dialogs[dialog_id] = datum
    print(f"# dialogs: {len(memory_dialogs)}")

    # Load image features.
    coco_features = np.load(
        args["input_feature_path"], allow_pickle=True, encoding="bytes"
    )[()]

    progress_bar = tqdm.tqdm(memory_dialogs.items(), desc="Extracting features")
    for dialog_id, datum in progress_bar:
        assert datum["memory_graph_id"] in memory_graphs, "Memory graph missing!"
        graph = memory_graphs[datum["memory_graph_id"]]
        sample_memories = {}
        for ii in graph["memories"]:
            if ii["memory_id"] in datum["mentioned_memory_ids"]:
                sample_memories[ii["memory_id"]] = ii

        for mem_id, mem_datum in sample_memories.items():
            image_id = mem_datum["media"][0]["media_id"]
            memory_feature = coco_features[image_id]
            feature_save_path = os.path.join(
                args["feature_save_path"], f"mscoco_butd_{image_id}.npy"
            )
            np.save(feature_save_path, memory_feature)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dialog_json", nargs="+", required=True, help="Input memories JSON"
    )
    parser.add_argument(
        "--input_memory_json", nargs="+", required=True, help="Input memories metadata"
    )
    parser.add_argument(
        "--feature_save_path", required=True, help="Folder to save memory features"
    )
    parser.add_argument(
        "--input_feature_path", required=True, help="Path to image features"
    )

    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
