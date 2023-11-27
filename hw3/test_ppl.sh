export CUDA_VISIBLE_DEVICES=8
python ppl.py \
    --base_model_path="./Taiwan-LLM-7B-v2.0-chat" \
    --peft_path="./adapter_checkpoint" \
    --test_data_path="./data/public_test.json"