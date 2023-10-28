export CUDA_VISIBLE_DEVICES=8

# --source_prefix='summarize: ' \ to let the model know that we are trying to do summarization
python run_summarization_no_trainer.py \
    --model_name_or_path='google/mt5-small' \
    --train_file='./data/train.jsonl' \
    --validation_file='./data/public.jsonl' \
    --source_prefix='summarize: ' \ 
    --text_column='maintext' \
    --summary_column='title' \
    --num_beams=3 \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --output_dir='./summarization-ckpt/