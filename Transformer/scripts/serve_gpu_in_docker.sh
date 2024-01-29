#!/bin/bash

if [ -z "${ASKCOS_REGISTRY}" ]; then
  export ASKCOS_REGISTRY=registry.gitlab.com/mlpds_mit/askcosv2
fi

docker run -d --rm --gpus '"device=0"' \
  --name forward_augmented_transformer \
  -p 9510-9512:9510-9512 \
  -v "$PWD/mars":/app/augmented_transformer/mars \
  -t "${ASKCOS_REGISTRY}"/forward_predictor/augmented_transformer:1.0-gpu \
  torchserve \
  --start \
  --foreground \
  --ncs \
  --model-store=/app/augmented_transformer/mars \
  --models \
  USPTO_480k_mix=USPTO_480k_mix.mar \
  --ts-config ./config.properties
