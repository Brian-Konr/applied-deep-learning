export CUDA_VISIBLE_DEVICES=8

python end_to_end_model.py \
    --train_file="./data/train.json" \
    --tokenizer_name="hfl/chinese-xlnet-base" \
    --validation_file="./data/valid.json" \
    --context_file="./data/context.json" \
    --model_name_or_path="hfl/chinese-xlnet-base" \
    --max_seq_length=1536 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_step=8 \
    --num_train_epochs=1 \
    --learning_rate=3e-4 \
    --output_dir="./end-to-end-ckpt-test"