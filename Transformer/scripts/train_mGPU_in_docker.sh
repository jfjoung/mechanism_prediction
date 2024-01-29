#!/bin/bash
export BATCH_SIZE=4096
export ACCUMULATION_COUNT=4
export NUM_NODES=1
export NUM_GPU=2

docker run --rm --shm-size=5gb --gpus '"device=0,1"' \
  -v "$PWD/logs":/app/augmented_transformer/logs \
  -v "$PWD/checkpoints":/app/augmented_transformer/checkpoints \
  -v "$PWD/results":/app/augmented_transformer/results \
  -v "$PROCESSED_DATA_PATH":/app/augmented_transformer/data/tmp_for_docker/processed \
  -v "$MODEL_PATH":/app/augmented_transformer/checkpoints/tmp_for_docker \
  -t "${ASKCOS_REGISTRY}"/forward_predictor/augmented_transformer:1.0-gpu \
  python at_trainer.py \
  -world_size $NUM_GPU \
  -gpu_ranks 0 1\
  --do_train \
  --data="do_not_change_this_hardcode" \
  --model_name="augmented_transformer" \
  --data_name="$DATA_NAME" \
  --log_file="augmented_transformer_train_$DATA_NAME" \
  --processed_data_path=/app/augmented_transformer/data/tmp_for_docker/processed \
  --model_path=/app/augmented_transformer/checkpoints/tmp_for_docker \
  -seed 42 \
  -save_checkpoint_steps 10000 \
  -keep_checkpoint 10 \
  -train_steps 500000 \
  -param_init 0 \
  -param_init_glorot \
  -max_generator_batches 32 \
  -batch_size $((BATCH_SIZE / NUM_NODES / NUM_GPU)) \
  -accum_count 4 \
  -batch_type tokens \
  -normalization tokens \
  -max_grad_norm 0 \
  -optim adam \
  -adam_beta1 0.9 \
  -adam_beta2 0.998 \
  -decay_method noam \
  -warmup_steps 8000 \
  -learning_rate 2 \
  -label_smoothing 0.0 \
  -report_every 1000 \
  -layers 4 \
  -rnn_size 512 \
  -word_vec_size 512 \
  -encoder_type transformer \
  -decoder_type transformer \
  -dropout 0.1 \
  -position_encoding \
  -share_embeddings \
  -global_attention general \
  -global_attention_function softmax \
  -self_attn_type scaled-dot \
  --heads 8 \
  -transformer_ff 2048
