# export CUDA_VISIBLE_DEVICES=1 # NVIDIA GeForce
# export CUDA_VISIBLE_DEVICES=0 # NVIDIA RTX A5000
export CUDA_VISIBLE_DEVICES=3 # Tesla K80 (GPU 1)

python question_answering.py \
    --train_file="./data/train.json" \
    --validation_file="./data/valid.json" \
    --context_file="./data/context.json" \
    --model_name_or_path="bert-base-chinese" \
    --max_seq_length=512 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --num_train_epochs=1 \
    --learning_rate=3e-5 \
    --output_dir="./qa-ckpt"