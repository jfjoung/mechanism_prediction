#!/bin/bash

export ASKCOS_REGISTRY=registry.gitlab.com/mlpds_mit/askcosv2

export DATA_NAME="USPTO_480k_mix"
export TRAIN_FILE=$PWD/data/USPTO_480k_mix/raw/raw_train.csv
export VAL_FILE=$PWD/data/USPTO_480k_mix/raw/raw_val.csv
export TEST_FILE=$PWD/data/USPTO_480k_mix/raw/raw_test.csv
export PROCESSED_DATA_PATH=$PWD/data/$DATA_NAME/processed
export MODEL_PATH=$PWD/checkpoints/$DATA_NAME
export CHECKPOINT_PATH=$MODEL_PATH/model.300000_29.pt

export EXTRA_FILES="\
models.zip,\
utils.zip,\
train.py,\
/app/graph2smiles/checkpoints/model.pt,\
/app/graph2smiles/data/processed/vocab.txt,\
predict.py\
"

zip models.zip models/*
zip utils.zip utils/*

docker run --rm \
  -v "$PROCESSED_DATA_PATH":/app/graph2smiles/data/processed \
  -v "$CHECKPOINT_PATH":/app/graph2smiles/checkpoints/model.pt \
  -v "$PWD/models.zip":/app/graph2smiles/models.zip \
  -v "$PWD/utils.zip":/app/graph2smiles/utils.zip \
  -v "$PWD/mars":/app/graph2smiles/mars \
  -t "${ASKCOS_REGISTRY}"/forward_predictor/graph2smiles:1.0-gpu \
  torch-model-archiver \
  --model-name=USPTO_480k_mix \
  --version=1.0 \
  --handler=/app/graph2smiles/handler.py \
  --extra-files="$EXTRA_FILES" \
  --export-path=/app/graph2smiles/mars \
  --force
