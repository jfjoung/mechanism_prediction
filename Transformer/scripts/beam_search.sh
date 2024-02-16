#!/bin/bash

export CHECKPOINT="model_step_1250000.pt"

python at_beam_searcher.py \
  --model_name="augmented_transformer" \
  --data_name="$DATA_NAME" \
  --log_file="augmented_transformer_predict_$DATA_NAME" \
  --processed_data_path="$PROCESSED_DATA_PATH" \
  --model_path="$MODEL_PATH" \
  --test_output_path="$TEST_OUTPUT_PATH" \
  --test_unseen_path="$TEST_UNSEEN_PATH" \
  --checkpoint=$CHECKPOINT \
  -batch_size 16 \
  -replace_unk \
  -max_length 200 \
  -beam_size 10 \
  -n_best 20 \
  -gpu 0 \
  -model "do_not_change_this_hardcode" \
  --src="do_not_change_this_hardcode"
