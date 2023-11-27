from transformers import BitsAndBytesConfig
import torch
import json
import random

data = json.load(open("data/train.json", "r"))
data_len = len(data)

def get_examples() -> list:
    # random sample 10 examples
    random_idx = []
    for i in range(2):
        random_idx.append(random.randint(0, data_len))
    examples = []
    for idx in random_idx:
        examples.append(data[idx])
    return examples


def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"
    # example_prompt = "你是精通文言文與中文翻譯的老師，以下是學生和老師之間的對話。你要對學生的問題提供有用、安全、詳細和禮貌的回答。以下是幾個例子：\n"
    # examples = get_examples()
    # for example in examples:
    #     example_prompt += f"STUDENT: {example['instruction']} \n TEACHER: {example['output']}\n\n"
    # current_prompt = f"STUDENT: {instruction} \n TEACHER: "
    # return example_prompt + current_prompt

def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

def add_prompt(example):
    return {'input': get_prompt(example['instruction'])}