#!/bin/bash

export EXTRA_FILES="\
/app/augmented_transformer/utils.py,\
/app/augmented_transformer/checkpoints/USPTO_480k_mix/model_step_500000.pt\
"

docker run --rm \
  -v "$PWD/checkpoints/USPTO_480k_mix/model_step_500000.pt":/app/augmented_transformer/checkpoints/USPTO_480k_mix/model_step_500000.pt \
  -v "$PWD/mars":/app/augmented_transformer/mars \
  -t "${ASKCOS_REGISTRY}"/forward_predictor/augmented_transformer:1.0-gpu \
  torch-model-archiver \
  --model-name=USPTO_480k_mix \
  --version=1.0 \
  --handler=/app/augmented_transformer/at_handler.py \
  --extra-files="$EXTRA_FILES" \
  --export-path=/app/augmented_transformer/mars \
  --force
