# accept 3 required arguments:
# ${1}: path to context.json
# ${2}: path to test.json
# ${3}: path to the output prediction file named prediction.csv

context_file=${1}
test_file=${2}
output_file=${3}

python mc_inference.py \
    --test_file="${test_file}" \
    --context_file="${context_file}" \
    --model_path="./best-mc-ckpt" \
    --output_file="./mc_result.json"

python qa_inference.py \
    --test_file="${test_file}" \
    --context_file="${context_file}" \
    --mc_result_file="./mc_result.json" \
    --model_path="./best-qa-ckpt" \
    --output_file="${output_file}"