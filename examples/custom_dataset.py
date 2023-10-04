# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets
import itertools

from llama_recipes.datasets.utils import Concatenator



# Define special tokens for instructions, context, and responses
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "[SYS]", "[/SYS]"
def tokenize_dialog(dialog, tokenizer):
    max_length = 1000  # Set the desired maximum token length
    input_text = f"{B_INST} {dialog['instruction']} {E_INST} {B_SYS} {dialog['context']} {E_SYS} {dialog['response']} "
    input_ids = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    return {
        'input_ids': input_ids['input_ids'][0],
        'attention_mask': input_ids['attention_mask'][0],
        'labels': input_ids['input_ids'][0].clone()  # Set labels to input_ids for language modeling
    }

def get_custom_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("databricks/databricks-dolly-15k", split=split)
    
    dataset_split = dataset_split.map(lambda x: tokenize_dialog(x, tokenizer), remove_columns=['instruction', 'context', 'response','category'])
    dataset = dataset.map(Concatenator(), batched=True)
    
    return dataset