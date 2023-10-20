python mc_inference.py \
    --test_file="./data/test.json" \
    --context_file="./data/context.json" \
    --model_path="./roberta-mc-ckpt" \
    --output_file="./tmp/mc_result.json"

python qa_inference.py \
    --test_file="./data/test.json" \
    --context_file="./data/context.json" \
    --mc_result_file="./tmp/mc_result.json" \
    --model_path="./bertbase-qa-ckpt" \
    --output_file="./tmp/predict.csv"