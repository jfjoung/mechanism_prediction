#!/bin/bash

export MODEL=graph2smiles
export BATCH_TYPE=tokens
export BATCH_SIZE=4096

python beam_search.py \
   --do_predict \
   --do_score \
   --data_name="$DATA_NAME" \
   --model="$MODEL" \
   --load_from="$CHECKPOINT" \
   --log_file="graph2smiles_predict_$DATA_NAME" \
   --processed_data_path="$PROCESSED_DATA_PATH" \
   --model_path="$MODEL_PATH" \
   --test_output_path="$TEST_OUTPUT_PATH" \
   --test_file="$TEST_FILE" \
   --batch_type="$BATCH_TYPE" \
   --predict_batch_size="$BATCH_SIZE" \
   --accumulation_count=4 \
   --num_cores="$NUM_CORES" \
   --beam_size=30 \
   --predict_min_len=1 \
   --predict_max_len=512 \
   --log_iter=100

