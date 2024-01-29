#!/bin/bash


 docker run --rm \
   -v "$PWD/logs":/app/graph2smiles/logs \
   -v "$PWD/checkpoints":/app/graph2smiles/checkpoints \
   -v "$PWD/results":/app/graph2smiles/results \
   -v "$TRAIN_FILE":/app/graph2smiles/data/tmp_for_docker/raw_train.csv \
   -v "$VAL_FILE":/app/graph2smiles/data/tmp_for_docker/raw_val.csv \
   -v "$TEST_FILE":/app/graph2smiles/data/tmp_for_docker/raw_test.csv \
   -v "$PROCESSED_DATA_PATH":/app/graph2smiles/data/tmp_for_docker/processed \
   -t "${ASKCOS_REGISTRY}"/forward_predictor/graph2smiles:1.0-gpu \
   python preprocess.py \
   --model_name="graph2smiles" \
   --data_name="$DATA_NAME" \
   --log_file="graph2smiles_preprocess_$DATA_NAME" \
   --train_file=/app/graph2smiles/data/tmp_for_docker/raw_train.csv \
   --val_file=/app/graph2smiles/data/tmp_for_docker/raw_val.csv \
   --test_file=/app/graph2smiles/data/tmp_for_docker/raw_test.csv \
   --processed_data_path=/app/graph2smiles/data/tmp_for_docker/processed \
   --num_cores="$NUM_CORES" \
   --seed=42

#python preprocess.py \
#  --model_name="graph2smiles" \
#  --data_name="$DATA_NAME" \
#  --log_file="graph2smiles_preprocess_$DATA_NAME" \
#  --train_file="$TRAIN_FILE" \
#  --val_file="$VAL_FILE" \
#  --test_file="$TEST_FILE" \
#  --processed_data_path="$PROCESSED_DATA_PATH" \
#  --num_cores="$NUM_CORES" \
#  --seed=42
