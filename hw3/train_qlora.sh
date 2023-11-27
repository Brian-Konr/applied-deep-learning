export CUDA_VISIBLE_DEVICES=7
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