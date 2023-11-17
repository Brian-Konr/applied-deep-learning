python run_summarization_no_trainer.py \
    --model_name_or_path='google/mt5-small' \
    --train_file='./data/train.jsonl' \
    --validation_file='./data/public.jsonl' \
    --text_column='maintext' \
    --summary_column='title' \
    --num_beams=3 \
    --num_train_epochs=10 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --output_dir='./summarization-ckpt' \
    --num_warmup_steps=300 \
    --learning_rate=5e-5 \
    --checkpointing_steps="epoch"