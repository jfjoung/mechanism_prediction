#!/bin/bash

export ASKCOS_REGISTRY=registry.gitlab.com/mlpds_mit/askcosv2
docker build -f Dockerfile_gpu -t ${ASKCOS_REGISTRY}/forward_predictor/graph2smiles:1.0-gpu .
bash scripts/benchmark_in_docker.sh 


