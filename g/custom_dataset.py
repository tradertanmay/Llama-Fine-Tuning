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
import torch
from torch.nn import functional as F
from itertools import chain




# Define your Concatenator class






def get_custom_dataset( dataset_config,tokenizer, split):
    

    ds = datasets.load_dataset("cais/mmlu","all")
    train_testvalid = ds['auxiliary_train'].train_test_split(test_size=0.95, seed =20)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.02, seed =20)
    ds = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test']})
    
    train_dataset=ds["train"]
    test_dataset = ds["test"]  
    tokenizer.pad_token = tokenizer.eos_token

    def format_dolly(sample):
        question = f"### quetion\n{sample['question']}"
        choices = f"### choices\n{sample['choices']}"
        answer = f"### answer\n{sample['answer']}"
        # join all the parts together
        prompt = "\n\n".join([i for i in [question, choices,answer] if i is not None])
        return prompt

    def template_dataset(sample):
        sample["text"] = f"{format_dolly(sample)}{tokenizer.eos_token}"
         
        return sample
        
        



    if split == "train":
        train_dataset = train_dataset.map(template_dataset, remove_columns=list(train_dataset.features))

        train_dataset = train_dataset.map(
                    lambda sample: tokenizer(sample["text"],padding='max_length', truncation=True, max_length=1000), batched=True)
        train_dataset = train_dataset.remove_columns("text")

        train_dataset = train_dataset.map(Concatenator(), batched=True)

        return train_dataset

        
    elif split == "validation":
            test_dataset = test_dataset.map(template_dataset, remove_columns=list(test_dataset.features))
            test_dataset = test_dataset.map(
                                lambda sample: tokenizer(sample["text"],padding='max_length', truncation=True, max_length=1000), batched=True)
            test_dataset = test_dataset.remove_columns("text")
        
            test_dataset = test_dataset.map(Concatenator(), batched=True)
              
            return test_dataset
    
