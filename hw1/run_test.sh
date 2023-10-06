export DATASET_NAME=swag

python multiple_choice.py \
  --model_name_or_path bert-base-cased \
  --dataset_name $DATASET_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$DATASET_NAME/ \
  --debug