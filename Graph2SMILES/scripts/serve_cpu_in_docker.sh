#!/bin/bash

if [ -z "${ASKCOS_REGISTRY}" ]; then
  export ASKCOS_REGISTRY=registry.gitlab.com/mlpds_mit/askcosv2
fi

docker run -d --rm \
  --name forward_graph2smiles \
  -p 9520-9522:9520-9522 \
  -v "$PWD/mars":/app/graph2smiles/mars \
  -t "${ASKCOS_REGISTRY}"/forward_predictor/graph2smiles:1.0-cpu \
  torchserve \
  --start \
  --foreground \
  --ncs \
  --model-store=/app/graph2smiles/mars \
  --models \
  USPTO_480k_mix=USPTO_480k_mix.mar \
  --ts-config ./config.properties
