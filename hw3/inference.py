import json
from model_helper import load_tw_llama_model_and_tokenizer
from peft import PeftModel
from utils import get_prompt
from transformers import GenerationConfig, pipeline
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test_file_path", type=str, default="data/private_test.json", help="Path to the test file.")
parser.add_argument("--tw_llama_model_path", type=str, default="./Taiwan-LLM-7B-v2.0-chat", help="Path to the Taiwan-LLM-7B-v2.0-chat model.")
parser.add_argument("--peft_model_path", type=str, default="./qlora-out-3/checkpoint-1000", help="Path to the PEFT model.")
parser.add_argument("--output_file_path", type=str, default="output/prediction.json", help="Path to the output file.")
parser.add_argument("--check_file_path", type=str, default="output/prediction_check.json", help="Path to the check file.")

args = parser.parse_args()

private_data = json.load(open(args.test_file_path, "r"))

tw_llama_model, tw_llama_tokenizer = load_tw_llama_model_and_tokenizer(args.tw_llama_model_path)
model = PeftModel.from_pretrained(tw_llama_model, args.peft_model_path)
model.eval()

check = []
output = []
for data in tqdm(private_data):
    input_seq = get_prompt(data["instruction"])
    inputs = tw_llama_tokenizer(input_seq, padding=True, return_tensors="pt").to('cuda')
    outputs = model.generate(
        **inputs, 
        generation_config=GenerationConfig(
            do_sample=True,
            max_new_tokens=512,
            top_p=0.99,
            temperature=1e-8,
        )
    )
    outputs = tw_llama_tokenizer.decode(outputs[0], skip_special_tokens=True).replace(input_seq, "").strip()
    check.append({
        "instruction": data["instruction"],
        "output": outputs
    })
    output.append({
        "id": data["id"],
        "output": outputs 
    })

with open(args.output_file_path, "w") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

with open(args.check_file_path, "w") as f:
    json.dump(check, f, indent=4, ensure_ascii=False)