# ADL 20234 Fall HW1
資工所 碩士一年級 郭子麟 R12922050

## Environment
- Python 3.9.18
- Packages: See requirements.txt

## How to Train
First, put the dataset in `./data/` folder:

- For Multiple Choice, run the following command: `sh multiple_choice_train.sh`
    - It will generate the model in `/mc-ckpt/` folder
- For Question Answering, run the following command: `sh question_answering_train.sh`
    - It will generate the model in `/qa-ckpt/` folder
    - If you want to train the Question Answering model from scratch, please run `question_answering_no_pretrain_train.sh`
- For End-to-End, run the following command: `sh end_to_end_train.sh`
    - It will generate the model in `/end-to-end-ckpt/` folder

## How to Test
- run the following command: `sh run.sh`
    - It requires three arguments: `context_json_path`, `test_json_path`, `output_csv_path`

## About Plot
- Code for plot is in `plot.ipynb`
    - It requires QA model to be trained first so that it will generate a `loss_em.json` file in `./qa-ckpt/` folder
