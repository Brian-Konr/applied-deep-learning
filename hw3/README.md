# ADL 20234 Fall HW3
資工所 碩士一年級 郭子麟 R12922050

## Environment
- Python 3.9.18
- Packages: See requirements.txt

## Training QLora
The main training script is `train_qlora.py`. You can run the script with the following command:
```bash
python train_qlora.py \
    --lora_r=4 \
    --lora_alpha=16 \
    --lora_dropout=0.1 \
    --num_train_epochs=1 \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=8 \
    --train_file="./data/train.json" \
    --lora_output_dir="./qlora-out" \
    --model_name_or_path="./Taiwan-LLM-7B-v2.0-chat"
```
I also provided a different way to train the model, which is to use p-tuning. You can run the p-tuning version by adding `--use_p_tuning=True` to the above command.

In this script, you can find several important components:

### `load_model_and_tokenizer`
This function loads the model and tokenizer from the pretrained model. The pretrained model is `Taiwan-LLM-7B-v2.0-chat` TAs provided in this case. You can find the method in `tw_llama_helper.py`.

### `create_peft_model`
This function creates a basic Lora model with the pretrained model. You can specify multiple hyper-parameters in this function. The method is in `lora_helper.py`.

### `DataCollator`
I created a customized `DataCollatorForCausalLM` class to handle the data collation. The class is in `DataCollatorForCausalLM.py`.

### `train`
I used `Trainer` class provided by `transformers` to train the model.

## Inference
The main inference script is `inference.py`. You can run the script with the following command:
```bash
bash run.sh ${1} ${2} ${3} ${4}
```
Meaning of the 4 arguments are given as follows:
- `${1}`: path to the Taiwan-LLaMa checkpoint folder
- `${2}`: path to the adapter_checkpoint downloaded under your folder
- `${3}`: path to the input file (.json)
- `${4}`: path to the output file (.json)

For example, you can run the following command to get the result:
```bash
bash run.sh Taiwan-LLM-7B-v2.0-chat/ adapter_checkpoint/ data/private_test.json prediction.json
```
