#!/bin/bash

python preprocess.py \
 --model_name="graph2smiles" \
 --data_name="$DATA_NAME" \
 --log_file="graph2smiles_preprocess_$DATA_NAME" \
 --train_file="$TRAIN_FILE" \
 --val_file="$VAL_FILE" \
 --test_file="$TEST_FILE" \
 --processed_data_path="$PROCESSED_DATA_PATH" \
 --num_cores="$NUM_CORES" \
 --seed=42
