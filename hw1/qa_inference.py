from transformers import pipeline
import torch
import time
import json
# construct parser
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser()
# needs two arguments, test.json file path and context.json file path
parser.add_argument("--test_file", type=str, default="./data/test.json", help="test file path")
parser.add_argument("--context_file", type=str, default="./data/context.json", help="context file path")
parser.add_argument("--mc_result_file", type=str, default="./results/mc_result.json", help="mc answer file path")
parser.add_argument("--model_path", type=str, default="./best-qa-ckpt/", help="model path")
parser.add_argument("--output_file", type=str, default="./results/predict.csv", help="output file path")


args = parser.parse_args()

# load context and turn it into a dictionary
if args.context_file is None:
    raise ValueError("Context file must be provided")
with open(args.context_file, "r") as f:
    context = json.load(f)

# load mc_result file
if args.mc_result_file is None:
    raise ValueError("MC result file must be provided")
with open(args.mc_result_file, "r") as f:
    mc_result = json.load(f)

# load test file
if args.test_file is None:
    raise ValueError("Test file must be provided")
with open(args.test_file, "r") as f:
    test = json.load(f)

# merge mc_result and test
mc_dict = {}
for entry in mc_result:
    mc_dict[entry["id"]] = entry["relevant"]

merged_test = []
for entry in test:
    entry["relevant"] = mc_dict[entry["id"]]
    merged_test.append(entry)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

qa_model = pipeline("question-answering", model=args.model_path, device=0)

print(f"start to inference for {len(test)} test cases")
# start to count time
start_time = time.time()

ans = []
for i in tqdm(range(len(merged_test)), desc="Inference progress"):
    question = merged_test[i]["question"]
    context_paragraph_id = merged_test[i]["relevant"]
    context_paragraph = context[context_paragraph_id]
    qa_answer = qa_model(question=question, context=context_paragraph)
    # if answer contains comma, use quotation mark to wrap it
    if "," in qa_answer["answer"]:
        qa_answer["answer"] = f"\"{qa_answer['answer']}\""
    ans.append({
        "id": merged_test[i]["id"],
        "answer": qa_answer["answer"]
    })

# save ans to predict.csv
# first row should be "id,answer"

with open(args.output_file, "w") as f:
    f.write("id,answer\n")
    for entry in ans:
        f.write(f"{entry['id']},{entry['answer']}\n")