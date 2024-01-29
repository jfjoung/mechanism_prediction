#!/bin/bash

singularity instance start --nv graph2smiles_gpu.sif forward_graph2smiles
nohup \
singularity exec --nv instance://forward_graph2smiles \
  torchserve \
  --start \
  --foreground \
  --ncs \
  --model-store=./mars \
  --models \
  USPTO_480k_mix=USPTO_480k_mix.mar \
  --ts-config ./config.properties \
&>/dev/null &
