#!/bin/bash

docker run --shm-size=5gb --rm --gpus '"device=0"' \
  -v "$PWD/logs":/app/augmented_transformer/logs \
  -v "$PWD/checkpoints":/app/augmented_transformer/checkpoints \
  -v "$PWD/results":/app/augmented_transformer/results \
  -v "$TEST_FILE":/app/augmented_transformer/data/tmp_for_docker/raw_test.csv \
  -v "$PROCESSED_DATA_PATH":/app/augmented_transformer/data/tmp_for_docker/processed \
  -v "$MODEL_PATH":/app/augmented_transformer/checkpoints/tmp_for_docker \
  -v "$TEST_OUTPUT_PATH":/app/augmented_transformer/results/tmp_for_docker \
  -t "${ASKCOS_REGISTRY}"/forward_predictor/augmented_transformer:1.0-gpu \
  python at_beam_searcher.py \
  --model_name="augmented_transformer" \
  --data_name="$DATA_NAME" \
  --log_file="augmented_transformer_predict_$DATA_NAME" \
  --test_file=/app/augmented_transformer/data/tmp_for_docker/raw_test.csv \
  --processed_data_path=/app/augmented_transformer/data/tmp_for_docker/processed \
  --model_path=/app/augmented_transformer/checkpoints/tmp_for_docker \
  --test_output_path=/app/augmented_transformer/results/tmp_for_docker \
  --test_unseen_name=$(basename "$TEST_FILE" | cut -d '.' -f 1) \
  --aug_factor=5 \
  -batch_size 16 \
  -replace_unk \
  -max_length 200 \
  -beam_size 10 \
  -n_best 20 \
  -gpu 0 \
  -model "do_not_change_this_hardcode" \
  --src="do_not_change_this_hardcode"