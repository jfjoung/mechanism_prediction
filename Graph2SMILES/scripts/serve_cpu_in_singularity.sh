#!/bin/bash

singularity instance start graph2smiles_cpu.sif forward_graph2smiles
nohup \
singularity exec instance://forward_graph2smiles \
  torchserve \
  --start \
  --foreground \
  --ncs \
  --model-store=./mars \
  --models \
  USPTO_480k_mix=USPTO_480k_mix.mar \
  --ts-config ./config.properties \
&>/dev/null &
