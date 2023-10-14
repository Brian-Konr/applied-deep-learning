from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice
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
parser.add_argument("--output_file", type=str, default="./results/mc_result.json", help="output file path")

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

tokenizer = AutoTokenizer.from_pretrained("./best-mc-ckpt/")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
model = AutoModelForMultipleChoice.from_pretrained("./best-mc-ckpt/").to(device)
# turn on evaluation mode for inference
model.eval()

# start to count time
start_time = time.time()

# start to inference for each test case
ans = []
print(f"start to inference for {len(test)} test cases")
for i in tqdm(range(len(test)), desc="Inference progress"):
    prompt = test[i]["question"]
    candidates = [context[paragraph_id] for paragraph_id in test[i]["paragraphs"]]
    inputs = tokenizer([[prompt, candidate] for candidate in candidates], max_length=512, return_tensors="pt", padding="max_length", truncation=True)
    labels = torch.tensor(0).unsqueeze(0).to(device)
    outputs = model(**{k: v.unsqueeze(0).to(device) for k, v in inputs.items()}, labels=labels)
    logits = outputs.logits
    predicted_class = logits.argmax().item()
    # save the result to mc_result.json, where each entry is a dictionary with the following format:
    # {"id": 0, "relevant": 0}
    ans.append({
        "id": test[i]["id"],
        "relevant": test[i]["paragraphs"][predicted_class]
    })
# save ans to mc_result.json
with open(args.output_file, "w") as f:
    json.dump(ans, f)
print("--- %s seconds ---" % (time.time() - start_time))
