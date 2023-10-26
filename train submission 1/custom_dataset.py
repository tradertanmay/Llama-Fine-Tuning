import copy
from datasets import load_dataset
from datasets import DatasetDict  # Import DatasetDict
from llama_recipes.datasets.utils import Concatenator
import torch
from datasets import load_dataset, Dataset 
import copy
import datasets
import itertools
from torch.utils.data import random_split



from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

from itertools import chain







def get_custom_dataset( dataset_config,tokenizer, split):
    

    # Load and tokenize the first dataset
    dataset1 = datasets.load_dataset("GAIR/lima","plain_text")
    dataset2 = dataset1["train"].train_test_split(test_size=0.1, seed =20)
    train_dataset = dataset2["train"]
    tokenizer.pad_token = tokenizer.eos_token
    
    test_dataset = dataset2["test"]

    

    def format_dolly(sample):
        instruction = f"### Instruction\n{sample['conversations'][0]}"
        response = f"### Answer\n{sample['conversations'][1]}"
        # join all the parts together
        prompt = "\n\n".join([i for i in [instruction, response] if i is not None])
        return prompt

    def template_dataset(sample):
        sample["text"] = f"{format_dolly(sample)}{tokenizer.eos_token}"
        return sample

    
    if split == "train":
        
        train_dataset = train_dataset.map(template_dataset, remove_columns=list(train_dataset.features))
        train_dataset = train_dataset.map(
            lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(train_dataset.features))
        #train_dataset = train_dataset.map(Concatenator(), batched=True)
        return train_dataset

    elif split == "validation":
        test_dataset = test_dataset.map(template_dataset, remove_columns=list(test_dataset.features))
        test_dataset = test_dataset.map(
            lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(test_dataset.features))
        #test_dataset = test_dataset.map(Concatenator(), batched=True)
        return test_dataset

       






