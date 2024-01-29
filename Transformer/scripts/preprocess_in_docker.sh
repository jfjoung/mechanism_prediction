#!/bin/bash

docker run --rm \
  -v "$PWD/logs":/app/augmented_transformer/logs \
  -v "$PWD/checkpoints":/app/augmented_transformer/checkpoints \
  -v "$PWD/results":/app/augmented_transformer/results \
  -v "$TRAIN_FILE":/app/augmented_transformer/data/tmp_for_docker/raw_train.csv \
  -v "$VAL_FILE":/app/augmented_transformer/data/tmp_for_docker/raw_val.csv \
  -v "$TEST_FILE":/app/augmented_transformer/data/tmp_for_docker/raw_test.csv \
  -v "$PROCESSED_DATA_PATH":/app/augmented_transformer/data/tmp_for_docker/processed \
  -t "${ASKCOS_REGISTRY}"/forward_predictor/augmented_transformer:1.0-gpu \
  python at_processor.py \
  --model_name="augmented_transformer" \
  --data_name="$DATA_NAME" \
  --log_file="augmented_transformer_preprocess_$DATA_NAME" \
  --train_file=/app/augmented_transformer/data/tmp_for_docker/raw_train.csv \
  --val_file=/app/augmented_transformer/data/tmp_for_docker/raw_val.csv \
  --test_file=/app/augmented_transformer/data/tmp_for_docker/raw_test.csv \
  --processed_data_path=/app/augmented_transformer/data/tmp_for_docker/processed \
  --aug_factor=5 \
  --num_cores="$NUM_CORES" \
  --seed=42 \
  --save_data="do_not_change_this_hardcode" \
  --train_src="do_not_change_this_hardcode" \
  --train_tgt="do_not_change_this_hardcode" \
  --valid_src="do_not_change_this_hardcode" \
  --valid_tgt="do_not_change_this_hardcode"
