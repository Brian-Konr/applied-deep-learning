import argparse, os
from datasets import load_dataset
from lora_helper import create_peft_model
from tw_llama_helper import load_model_and_tokenizer
from peft.tuners.lora import LoraLayer
from utils import add_prompt, get_prompt
from DataCollatorForCausalLM import DataCollatorForCausalLM
from transformers import (
    Trainer,
    TrainingArguments,
)

parser = argparse.ArgumentParser()

parser.add_argument('--lora_r', type=int, default=32)
parser.add_argument('--lora_alpha', type=int, default=16)
parser.add_argument('--lora_dropout', type=float, default=0.05)
parser.add_argument('--train_file', type=str, default="./data/train.json")
parser.add_argument('--lora_output_dir', type=str, default="./qlora-out")
parser.add_argument('--num_train_epochs', type=int, default=4)
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
parser.add_argument('--per_device_train_batch_size', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=2e-4)

args = parser.parse_args()

# create a directory
if not os.path.exists(args.lora_output_dir):
    os.makedirs(args.lora_output_dir)

# record the arguments into a txt file
with open(f"{args.lora_output_dir}/args.txt", "w") as f:
    for key, value in vars(args).items():
        f.write(f"{key}: {value}\n")
    prompt = get_prompt("\{instruction\}")
    f.write(f"prompt: {prompt}\n")

# load base model
tw_llama_model, tw_llama_tokenizer = load_model_and_tokenizer("./Taiwan-LLM-7B-v2.0-chat")

# construct peft model
peft_model = create_peft_model(
    tw_llama_model, 
    lora_r=args.lora_r, 
    lora_alpha=args.lora_alpha, 
    lora_dropout=args.lora_dropout,
    bf16=False
)

# dataset = load_dataset("json", data_files="./data/train_modified.json")
dataset = load_dataset("json", data_files=args.train_file)['train']
dataset = dataset.map(add_prompt)

data_collator = DataCollatorForCausalLM(
    tokenizer=tw_llama_tokenizer,
    source_max_len=280, 
    target_max_len=512,
    train_on_source=False,
    predict_with_generate=False
)

# define training args
output_dir = args.lora_output_dir
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=args.per_device_train_batch_size,
    learning_rate=args.learning_rate,
    num_train_epochs=args.num_train_epochs,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    gradient_checkpointing=False,
    # logging strategies
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=10,
    remove_unused_columns=False,
    bf16=False,
)

# define trainer
trainer = Trainer(  
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# train
trainer.train()

# save model
trainer.save_model(output_dir)