# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets
import itertools

from llama_recipes.datasets.utils import Concatenator



# Define special tokens for instructions, context, and responses
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
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
def tokenize_dialog1(dialog, tokenizer):
    dialog_tokens = [
            tokenizer(
                f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
            )
            for prompt, answer in zip(dialog[::2], dialog[1::2])
        ]
    if len(dialog) % 2:    
        dialog_tokens += [tokenizer(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        )]
    
    combined_tokens = {}  
    for k in dialog_tokens[0].keys():
        combined_tokens[k] = list(itertools.chain(*(t[k] for t in dialog_tokens)))
    return combined_tokens
    
def get_custom_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("databricks/databricks-dolly-15k", split=split)
    
    dataset_split = dataset_split.map(lambda x: tokenize_dialog(x, tokenizer), remove_columns=['instruction', 'context', 'response','category'])
    dataset = dataset.map(Concatenator(), batched=True)

    dataset1 = datasets.load_dataset("OpenAssistant/oasst1", split=split)
    
    dataset1 = dataset1.map(lambda sample: {
        "message_id": sample["message_id"],
        "parent_id": sample["parent_id"],
        "text": sample["text"],
        },
        batched=True,
        remove_columns=list(dataset1.features),)
    
    nodes = {}
    
    messages = {}
    root_ids = []
    
    for data in dataset1:
        if data["parent_id"]:
            nodes[data["parent_id"]] = nodes.get(data["parent_id"], []) + [data["message_id"]]
        else:
            root_ids.append(data["message_id"])
        messages[data["message_id"]]=data["text"]
           
    def follow(thread, current_id):
        thread = copy.copy(thread) + [messages[current_id]]
        if current_id in nodes:
            new_threads = []
            for next_id in nodes[current_id]:
                new_threads += follow(thread, next_id)
            return new_threads
        else:
            return [thread]
        
    def get_threads_from_root(root_id):
        all_threads = []
        thread = [messages[root_id]]
        for cid in nodes[root_id]:
            all_threads += follow(thread, cid)
        return all_threads
            
    dataset1 = dataset1.filter(lambda x: x["message_id"] in root_ids)
    dataset1 = dataset1.map(lambda x: {"thread": get_threads_from_root(x["message_id"])}, remove_columns=list(dataset1.features))
    dataset1 = dataset1.map(lambda x: {"thread": [i for row in x["thread"] for i in row]}, batched=True)
    
    def to_dialog(thread):
        dialog = []
        for i, content in enumerate(thread):
            dialog.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": content,
            })
        return {"dialog": dialog}
            
    dataset1 = dataset1.map(lambda x: to_dialog(x["thread"]), remove_columns=list(dataset1.features))
    dataset1 = dataset1.map(lambda x: tokenize_dialog1(x["dialog"], tokenizer), remove_columns=list(dataset.features))
    dataset1 = dataset1.map(Concatenator(), batched=True)
    
    # Concatenate the two datasets
    combined_dataset = concatenate_datasets([dataset, dataset1])
    
    return combined_dataset
