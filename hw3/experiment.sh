export CUDA_VISIBLE_DEVICES=7

qlora_ver=$1
checkpoint=$2

echo "Start Running QLora Ver $qlora_ver checkpoint $checkpoint"
python ppl.py \
    --base_model_path="./Taiwan-LLM-7B-v2.0-chat" \
    --peft_path="./qlora-out-$qlora_ver/checkpoint-$checkpoint" \
    --test_data_path="./data/public_test.json"
echo "Finished QLora Ver $qlora_ver checkpoint $checkpoint"