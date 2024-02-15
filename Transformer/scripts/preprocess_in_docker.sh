#!/bin/bash

python at_processor.py \
  --model_name="augmented_transformer" \
  --data_name="$DATA_NAME" \
  --log_file="augmented_transformer_preprocess_$DATA_NAME" \
  --train_file="$TRAIN_FILE" \
  --val_file="$VAL_FILE" \
  --test_file="$TEST_FILE" \
  --processed_data_path="$PROCESSED_DATA_PATH" \
  --aug_factor=1 \
  --num_cores="$NUM_CORES" \
  --seed=42 \
  --save_data="do_not_change_this_hardcode" \
  --train_src="do_not_change_this_hardcode" \
  --train_tgt="do_not_change_this_hardcode" \
  --valid_src="do_not_change_this_hardcode" \
  --valid_tgt="do_not_change_this_hardcode"
