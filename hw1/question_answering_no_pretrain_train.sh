export CUDA_VISIBLE_DEVICES=8

python question_answering.py \
    --train_file="./data/train.json" \
    --validation_file="./data/valid.json" \
    --context_file="./data/context.json" \
    --model_type="bert" \
    --tokenizer_name="bert-base-chinese" \
    --max_seq_length=512 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=2 \
    --num_train_epochs=1 \
    --learning_rate=3e-5 \
    --output_dir="./no-pretrain-qa-ckpt"