# coding: utf-8
"""Dataset Loader for Memory Dialogs.

Author(s): noctli, skottur
"""

import json
import pickle
import logging
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
from itertools import chain

from dataset import tokenize


# from train import SPECIAL_TOKENS, MODEL_INPUTS, PADDED_INPUTS
SPECIAL_TOKENS = ["<bos>", "<eos>", "<user>", "<system>", "<video>", "<pad>"]
SPECIAL_TOKENS_DICT = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "additional_special_tokens": ["<user>", "<system>", "<video>", "<cap>"],
    "pad_token": "<pad>",
}
MODEL_INPUTS = ["input_ids", "token_type_ids", "lm_labels"]
PADDED_INPUTS = ["input_ids", "token_type_ids", "lm_labels"]


def get_dataset(tokenizer, data_file, feature_path=None, n_history=3):
    """Get dataset given tokenizer and data file.
    """
    with open(data_file, "r") as file_id:
        dialog_data = json.load(file_id)

    dialog_list = []
    dialog_id_set = set()
    for dialog in dialog_data["dialogue_data"]:
        user_utterances = [
            tokenize(ii["transcript"], tokenizer) for ii in dialog["dialogue"]
        ]
        system_utterances = [
            tokenize(ii["system_transcript"], tokenizer) for ii in dialog["dialogue"]
        ]
        dialog_id = dialog["dialogue_idx"]
        dialog_id_set.add(dialog_id)

        num_turns = len(user_utterances)
        qalist = []
        history = []
        for turn_id in range(num_turns):
            user = user_utterances[turn_id]
            system = system_utterances[turn_id]
            history.append(user)
            if n_history == 0:
                dialog_list.append(
                    {"dialog_id": dialog_id, "history": [user], "response": system}
                )
            else:
                dialog_list.append(
                    {"dialog_id": dialog_id, "history": history, "response": system}
                )
            qalist.append(user)
            qalist.append(system)
            history = qalist[max(-len(qalist), -n_history * 2) :]

    all_features = {}
    # Ignore features for now.
    # if feature_path is not None:
    #     fea_types = ['vggish', 'i3d_flow', 'i3d_rgb']
    #     dataname = '<FeaType>/<ImageID>.npy'
    #     for ftype in fea_types:
    #         basename = dataname.replace('<FeaType>', ftype)
    #         features = {}
    #         for dialog_id in vid_set:
    #             filename = basename.replace('<ImageID>', vid)
    #             filepath = feature_path + filename
    #             features[vid] = (filepath, filepath)
    #         all_features[ftype] = features
    #     return dialog_list, all_features
    return dialog_list, None


class MemoryDialogDataset(Dataset):
    def __init__(self, dialogs, tokenizer, features=None, drop_rate=0.5, train=True):
        self.dialogs = dialogs
        self.features = features
        self.tokenizer = tokenizer
        self.drop_rate = drop_rate
        self.train = train

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        dialog = self.dialogs[index]
        dialog_id = dialog["dialog_id"]
        history = self.dialogs[index]["history"]
        response = self.dialogs[index]["response"]

        # if np.random.rand() < self.drop_rate:
        instance, _ = build_input_from_segments(
            history, response, self.tokenizer, video=False, train=self.train
        )
        # else:
        #     instance, _ = build_input_from_segments(
        #         history, response, self.tokenizer, video=False, train=self.train
        #     )
        input_ids = torch.Tensor(instance["input_ids"]).long()
        token_type_ids = torch.Tensor(instance["token_type_ids"]).long()
        lm_labels = torch.Tensor(instance["lm_labels"]).long()

        # if self.features is not None:
        #     try:
        #         vgg = np.load(self.features[0]["vggish"][vid][0])
        #         i3d_flow = np.load(self.features[0]["i3d_flow"][vid][0])
        #         i3d_rgb = np.load(self.features[0]["i3d_rgb"][vid][0])
        #     except KeyError:
        #         vgg = np.load(self.features[1]["vggish"][vid][0])
        #         i3d_flow = np.load(self.features[1]["i3d_flow"][vid][0])
        #         i3d_rgb = np.load(self.features[1]["i3d_rgb"][vid][0])

        #     sample_i3d_flow = i3d_flow[range(1, i3d_flow.shape[0], 1)]
        #     sample_i3d_rgb = i3d_rgb[range(1, i3d_rgb.shape[0], 1)]

        #     vgg = torch.from_numpy(vgg).float()
        #     i3d_flow = torch.from_numpy(sample_i3d_flow).float()
        #     i3d_rgb = torch.from_numpy(sample_i3d_rgb).float()
        #     min_length = min([i3d_flow.size(0), i3d_rgb.size(0), vgg.size(0)])
        #     i3d = torch.cat([i3d_flow[:min_length], i3d_rgb[:min_length], vgg[:min_length]], dim=1)

        #     return input_ids, token_type_ids, lm_labels, i3d
        # else:
        #     return input_ids, token_type_ids, lm_labels
        return input_ids, token_type_ids, lm_labels


def collate_fn(batch, pad_token, features=None):
    def padding(seq, pad_token):
        max_len = max([i.size(0) for i in seq])
        if len(seq[0].size()) == 1:
            result = torch.ones((len(seq), max_len)).long() * pad_token
        else:
            result = torch.ones((len(seq), max_len, seq[0].size(-1))).float()
        for i in range(len(seq)):
            result[i, : seq[i].size(0)] = seq[i]
        return result

    input_ids_list, token_type_ids_list, lm_labels_list, i3d_list = [], [], [], []
    for i in batch:
        input_ids_list.append(i[0])
        token_type_ids_list.append(i[1])
        lm_labels_list.append(i[2])
        if features is not None:
            i3d_list.append(i[3])

    input_ids = padding(input_ids_list, pad_token)
    token_type_ids = padding(token_type_ids_list, pad_token)
    lm_labels = padding(lm_labels_list, -1)
    input_mask = input_ids != pad_token
    if features is not None:
        i3d = padding(i3d_list, pad_token)
        i3d_mask = torch.sum(i3d != 1, dim=2) != 0
        input_mask = torch.cat([i3d_mask, input_mask], dim=1)
        i3d_labels = torch.ones((i3d.size(0), i3d.size(1))).long() * -1
        video_mask = torch.cat(
            [torch.zeros((i3d.size(0), i3d.size(1))), torch.ones(lm_labels.size())], 1
        )
        response_mask = torch.zeros(video_mask.size())
        lm_labels = torch.cat([i3d_labels, lm_labels], dim=1)
        return (
            input_ids,
            token_type_ids,
            lm_labels,
            input_mask,
            i3d,
            video_mask,
            response_mask,
        )
    else:
        return input_ids, token_type_ids, lm_labels, input_mask


def pad_dataset(dataset, padding=0):
    """ Pad the dataset.
    This could be optimized by defining a Dataset class and pad only 
    batches but this is simpler.
    """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [
            x + [padding if name != "labels" else -1] * (max_l - len(x))
            for x in dataset[name]
        ]
    return dataset


def build_input_from_segments(
    history, response, tokenizer, with_eos=True, video=False, train=True
):
    """ Build a sequence of input from 2 segments: history and last response """
    bos, eos, user, system = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-2])
    instance = {}
    sequence = history + [response + ([eos] if with_eos else [])]
    sequence = [
        [system if (len(sequence) - ii) % 2 else user] + ss
        for ii, ss in enumerate(sequence)
    ]

    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [
        system if ii % 2 else user for ii, ss in enumerate(sequence) for _ in ss
    ]
    if video:
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + sequence[
            -1
        ]
    else:
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + sequence[
            -1
        ]

    return instance, sequence
