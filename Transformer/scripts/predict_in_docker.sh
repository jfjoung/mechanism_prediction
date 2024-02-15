#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python at_predictor.py \
  --model_name="augmented_transformer" \
  --data_name="$DATA_NAME" \
  --log_file="augmented_transformer_predict_$DATA_NAME" \
  --test_file="$TEST_FILE" \
  --processed_data_path="$PROCESSED_DATA_PATH" \
  --model_path="$MODEL_PATH" \
  --test_output_path="$TEST_OUTPUT_PATH" \
  --aug_factor=1 \
  -batch_size 16 \
  -replace_unk \
  -max_length 200 \
  -beam_size 20 \
  -n_best 20 \
  -gpu 0 \
  -model "do_not_change_this_hardcode" \
  --src="do_not_change_this_hardcode"
