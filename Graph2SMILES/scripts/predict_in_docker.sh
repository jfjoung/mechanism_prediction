#!/bin/bash

# export CHECKPOINT="model.850000_84.pt"
export CHECKPOINT="model.620000_61.pt"
export MODEL=graph2smiles
export BATCH_TYPE=tokens
export BATCH_SIZE=4096

docker run --rm --gpus '"device=0"' \
   -v "$PWD/logs":/app/graph2smiles/logs \
   -v "$PWD/checkpoints":/app/graph2smiles/checkpoints \
   -v "$PWD/results":/app/graph2smiles/results \
   -v "$PROCESSED_DATA_PATH":/app/graph2smiles/data/tmp_for_docker/processed \
   -v "$MODEL_PATH":/app/graph2smiles/checkpoints/tmp_for_docker \
   -v "$TEST_OUTPUT_PATH":/app/graph2smiles/results/tmp_for_docker \
   -t "${ASKCOS_REGISTRY}"/forward_predictor/graph2smiles:1.0-gpu \
   python predict2.py \
   --do_predict \
   --do_score \
   --data_name="$DATA_NAME" \
   --model="$MODEL" \
   --load_from="$CHECKPOINT" \
   --log_file="graph2smiles_predict_$DATA_NAME" \
   --processed_data_path=/app/graph2smiles/data/tmp_for_docker/processed \
   --model_path=/app/graph2smiles/checkpoints/tmp_for_docker \
   --test_output_path=/app/graph2smiles/results/tmp_for_docker \
   --batch_type="$BATCH_TYPE" \
   --predict_batch_size="$BATCH_SIZE" \
   --accumulation_count=4 \
   --num_cores="$NUM_CORES" \
   --beam_size=30 \
   --predict_min_len=1 \
   --predict_max_len=512 \
   --log_iter=100

#python predict.py \
#  --do_predict \
#  --do_score \
#  --data_name="$DATA_NAME" \
#  --model="$MODEL" \
#  --load_from="$CHECKPOINT" \
#  --log_file="graph2smiles_predict_$DATA_NAME" \
#  --processed_data_path="$PROCESSED_DATA_PATH" \
#  --model_path="$MODEL_PATH" \
#  --test_output_path="$TEST_OUTPUT_PATH" \
#  --batch_type="$BATCH_TYPE" \
#  --predict_batch_size="$BATCH_SIZE" \
#  --accumulation_count=4 \
#  --num_cores="$NUM_CORES" \
#  --beam_size=30 \
#  --predict_min_len=1 \
#  --predict_max_len=512 \
#  --log_iter=100

