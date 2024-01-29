#!/bin/bash

singularity instance start augmented_transformer_cpu.sif forward_augmented_transformer
nohup \
singularity exec instance://forward_augmented_transformer \
  torchserve \
  --start \
  --foreground \
  --ncs \
  --model-store=./mars \
  --models \
  USPTO_480k_mix=USPTO_480k_mix.mar \
  --ts-config ./config.properties \
&>/dev/null &
