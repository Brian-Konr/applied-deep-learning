python inference_summarization.py \
    --model_path='./model' \
    --cuda_max_split_size=512 \
    --input_file=$1 \
    --output_file=$2 \
    --cuda_device=8 \
    --num_beams=3