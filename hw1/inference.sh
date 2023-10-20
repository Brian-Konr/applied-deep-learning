export CUDA_VISIBLE_DEVICES=3

python mc_inference.py \
    --test_file="./data/test.json" \
    --context_file="./data/context.json" \
    --model_path="./best-mc-ckpt" \
    --output_file="./results/mc_result.json"

python qa_inference.py \
    --test_file="./data/test.json" \
    --context_file="./data/context.json" \
    --mc_result_file="./results/mc_result.json" \
    --model_path="./best-qa-ckpt" \
    --output_file="./results/predict.csv"