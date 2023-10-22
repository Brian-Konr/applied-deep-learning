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
parser.add_argument("--model_path", type=str, help="model path")
parser.add_argument("--output_file", type=str, default="./results/predict.csv", help="output file path")


args = parser.parse_args()

# load context and turn it into a dictionary
if args.context_file is None:
    raise ValueError("Context file must be provided")
with open(args.context_file, "r") as f:
    context = json.load(f)

# load test file
if args.test_file is None:
    raise ValueError("Test file must be provided")
with open(args.test_file, "r") as f:
    test = json.load(f)

if args.model_path is None:
    raise ValueError("Model path must be provided")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

qa_model = pipeline("question-answering", model=args.model_path, device=0)

print(f"start to inference for {len(test)} test cases")
# start to count time
start_time = time.time()

ans = []
for i in tqdm(range(len(test)), desc="Inference progress"):
    question = test[i]["question"]
    context_str = ""
    for context_paragraph_id in test[i]["paragraphs"]:
        context_str += context[context_paragraph_id]
    qa_answer = qa_model(question=question, context=context_str)
    # if answer contains comma, use quotation mark to wrap it
    if "," in qa_answer["answer"]:
        qa_answer["answer"] = f"\"{qa_answer['answer']}\""
    # if answer is empty, use quotation mark to wrap it
    if qa_answer["answer"] == "":
        qa_answer["answer"] = "\"\""
    ans.append({
        "id": test[i]["id"],
        "answer": qa_answer["answer"]
    })

# save ans to predict.csv
# first row should be "id,answer"

with open(args.output_file, "w") as f:
    f.write("id,answer\n")
    for entry in ans:
        f.write(f"{entry['id']},{entry['answer']}\n")