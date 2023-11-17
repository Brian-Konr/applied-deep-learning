python inference_summarization.py \
    --model_path='./summarization-ckpt/batch-1-acc-2' \
    --cuda_max_split_size=512 \
    --input_file=$1 \
    --output_file=$2 \
    --cuda_device=8
python eval.py -r $1 -s $2
# sh run.sh data/public.jsonl results/submission.jsonl