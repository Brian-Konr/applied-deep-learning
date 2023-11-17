# ADL 20234 Fall HW2
資工所 碩士一年級 郭子麟 R12922050

## Environment
- Python 3.9.18
- Packages: See requirements.txt

## How to Train
First, put the dataset in `./data/` folder:

For training, run the following command: `sh train.sh`  
It will generate the model in `./summarization-ckpt/` folder

## How to Test
Run the following command: `sh run.sh`  
It requires two arguments:
  - `${1}`: path to the input file, e.g., `./data/public.jsonl`
  - `${2}`: path to the output file, e.g., `./output/output.jsonl`

## About Plot
Code and Data used for plotting the learning curve are in `draw` folder
  - each epoch's rouge score is recorded in `draw/epoch-results` folder
  - `draw.ipynb` is the code for plotting the learning curve
