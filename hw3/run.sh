export CUDA_VISIBLE_DEVICES=7
python inference.py \
    --tw_llama_model_path=$1 \
    --peft_model_path=$2 \
    --test_file_path=$3 \
    --output_file_path=$4