export CUDA_VISIBLE_DEVICES=8

python end_to_end_inference.py \
    --test_file="./data/test.json" \
    --context_file="./data/context.json" \
    --model_path="./end-to-end-ckpt" \
    --output_file="./results/e2e_predict.csv"