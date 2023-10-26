# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from tqdm import tqdm
from itertools import chain

from torch.utils.data import Dataset

import torch.nn.functional
from torch.utils.data import Dataset
import torch

from torch.utils.data import Dataset
from tqdm import tqdm

class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.samples = []

        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            input_ids = sample["input_ids"]
            attention_mask = sample["attention_mask"]
            labels = sample["labels"]

            # Extend the buffer lists with the current sample data
            buffer["input_ids"].extend(input_ids)
            buffer["attention_mask"].extend(attention_mask)
            buffer["labels"].append(labels)  # Assuming labels is a single integer for each sample

            # Check if the buffer size exceeds the chunk size, then create a chunk
            while len(buffer["input_ids"]) >= self.chunk_size:
                chunk = {
                    "input_ids": buffer["input_ids"][:self.chunk_size],
                    "attention_mask": buffer["attention_mask"][:self.chunk_size],
                    "labels": buffer["labels"][:self.chunk_size],
                }

                # Append the chunk to samples
                self.samples.append(chunk)

                # Remove processed data from buffer
                buffer["input_ids"] = buffer["input_ids"][self.chunk_size:]
                buffer["attention_mask"] = buffer["attention_mask"][self.chunk_size:]
                buffer["labels"] = buffer["labels"][self.chunk_size:]

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)
