from transformers import pipeline
from tqdm import tqdm
import pandas as pd
import argparse
import torch
import json
import os
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="./best-sum-ckpt", help="model path")
parser.add_argument("--input_file", type=str, default="./data/public.jsonl", help="input file path")
parser.add_argument("--output_file", type=str, default="./results/summarization_result.json", help="output file path")
parser.add_argument("--cuda_device", type=int, default=0, help="cuda device id")
parser.add_argument("--cuda_max_split_size", type=int, default=None, help="cuda max split size. default is None")
parser.add_argument("--do_sample", type=bool, default=False, help="Whether or not to use sampling ; use greedy decoding otherwise.")
parser.add_argument("--top_k", type=int, default=None, help="The number of highest probability vocabulary tokens to keep for top-k-filtering.")
parser.add_argument("--top_p", type=float, default=None, help="If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.")
parser.add_argument("--temperature", type=float, default=1.0, help="The value used to module the next token probabilities.")
parser.add_argument("--num_beams", type=int, default=1, help="The number of beams to use for beam search. 1 means no beam search.")


args = parser.parse_args()

if args.cuda_max_split_size is not None:
    # try set cuda env variable to avoid OOM    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{args.cuda_max_split_size}"

# load input file
input_df = pd.read_json(args.input_file, lines=True)

# load model
# check if cuda device id is valid oridnal
cuda_device = args.cuda_device
if args.cuda_device >= torch.cuda.device_count():
    print(f"cuda device id {cuda_device} is not valid!, use cuda 0 instead")
    cuda_device = 0

# conditionally add passed arguments to pipeline
generation_config = {}
if args.do_sample:
    generation_config["do_sample"] = args.do_sample
if args.top_k is not None:
    generation_config["top_k"] = args.top_k
if args.top_p is not None:
    generation_config["top_p"] = args.top_p
if args.temperature is not None:
    generation_config["temperature"] = args.temperature
if args.num_beams is not None:
    generation_config["num_beams"] = args.num_beams

print(f"generation_config: {generation_config}")
summarizer = pipeline(
    "summarization", 
    model=args.model_path, 
    device=cuda_device,
    **generation_config
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

results = []

for i in tqdm(range(len(input_df))):
    input_text = input_df.iloc[i]["maintext"]
    input_id = str(input_df.iloc[i]["id"])
    summary_text = summarizer(input_text)[0]["summary_text"]
    results.append({"title": summary_text, "id": input_id})

# save results to jsonl where each line is a json object with "title" and "id"
with open(args.output_file, "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

