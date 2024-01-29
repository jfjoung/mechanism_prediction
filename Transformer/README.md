# Augmented Transformer

Benchmarking and serving modules for reaction outcome prediction with Augmented Transformer, reimplemented using OpenNMT based on the manuscript (https://www.nature.com/articles/s41467-020-19266-y).

## Serving

### Step 1/4: Environment Setup

First set up the url to the remote registry
```
export ASKCOS_REGISTRY=registry.gitlab.com/mlpds_mit/askcosv2
```
Then follow the instructions below to use either Docker, or Singularity (if Docker or root privilege is not available). Building or pulling either CPU or GPU image would suffice. If GPUs are not available, just go with the CPU image.

#### Using Docker

- Option 1: pull pre-built image (ASKCOSv2 permission required)
```
(CPU) docker pull ${ASKCOS_REGISTRY}/forward_predictor/augmented_transformer:1.0-cpu
(GPU) docker pull ${ASKCOS_REGISTRY}/forward_predictor/augmented_transformer:1.0-gpu
```
- Option 2: build from local
```
(CPU) docker build -f Dockerfile_cpu -t ${ASKCOS_REGISTRY}/forward_predictor/augmented_transformer:1.0-cpu .
(GPU) docker build -f Dockerfile_gpu -t ${ASKCOS_REGISTRY}/forward_predictor/augmented_transformer:1.0-gpu .
```

#### Using Singularity

- Only option: build from local
```
(CPU) singularity build -f augmented_transformer_cpu.sif singularity_cpu.def
(GPU) singularity build -f augmented_transformer_gpu.sif singularity_gpu.def
```

### Step 2/4: Download Trained Models

```
sh scripts/download_trained_models.sh
```

### Step 3/4: Start the Service

#### Using Docker

```
(CPU) sh scripts/serve_cpu_in_docker.sh
(GPU) sh scripts/serve_gpu_in_docker.sh
```
 GPU-based container requires a CUDA-enabled GPU and the <a href="https://www.example.com/my great page">NVIDIA Container Toolkit</a> (or nvidia-docker in the past). By default, the first GPU will be used.

#### Using Singularity

```
(CPU) sh scripts/serve_cpu_in_singularity.sh
(GPU) sh scripts/serve_gpu_in_singularity.sh
```
The error messages related to torchserve logging can be safely ignored. Note that these scripts start the service in the background (i.e., in detached mode). So they would need to be explicitly stopped if no longer in use
```
(Docker)        docker stop forward_augmented_transformer
(Singularity)   singularity instance stop forward_augmented_transformer
```

### Step 4/4: Query the Service

- Sample query
```
curl http://0.0.0.0:9510/predictions/USPTO_480k_mix \
    --header "Content-Type: application/json" \
    --request POST \
    --data '{"smiles": ["[CH2:23]1[O:24][CH2:25][CH2:26][CH2:27]1.[F:1][c:2]1[c:3]([N+:10](=[O:11])[O-:12])[cH:4][c:5]([F:9])[c:6]([F:8])[cH:7]1.[H-:22].[NH2:13][c:14]1[s:15][cH:16][cH:17][c:18]1[C:19]#[N:20].[Na+:21]"]}'
```
- Sample response
```
List of
{
    "products": List[str],
    "scores": List[float]
},
```

### Unit Test for Serving (Optional)

Requirement: `requests` and `pytest` libraries (pip installable)

With the service started, run
```
pytest
```

## Benchmarking (GPU Required)

### Step 1/4: Environment Setup
Follow the instructions in Step 1 in the Serving section to build or pull the GPU docker image. It should have the name `${ASKCOS_REGISTRY}/forward_predictor/augmented_transformer:1.0-gpu`

Note: the Docker needs to be rebuilt before running whenever there is any change in code.

### Step 2/4: Data Preparation

Prepare the raw .csv files for train, validation and test (atom mapping not required). The required columns are "id" and "rxn_smiles", where "rxn_smiles" contains atom-mapped reaction SMILES, optionally with reagents.

### Step 3/4: Path Configuration

Configure the environment variables in ./scripts/benchmark_in_docker.sh, especially the paths, to point to the *absolute* paths of raw files and desired output paths.
```
export DATA_NAME="USPTO_480k_mix"
export TRAIN_FILE=$PWD/data/USPTO_480k_mix/raw/raw_train.csv
export VAL_FILE=$PWD/data/USPTO_480k_mix/raw/raw_val.csv
export TEST_FILE=$PWD/data/USPTO_480k_mix/raw/raw_test.csv
...
```

### Step 4/4: Benchmarking

Run benchmarking on a machine with GPU using
```
bash scripts/benchmark_in_docker.sh
```
This will run the preprocessing, training and predicting for Augmented Transformer with Top-n accuracies up to n=20 as the final outputs. Progress and result logs will be saved under ./logs.

The estimated running times for benchmarking the USPTO_480k dataset on a 32-core machine with 1 RTX3090 GPU are
* Preprocessing: ~1 hr
* Training: ~40 hrs
* Testing: ~30 mins

## Converting Trained Model into Servable Archive (Optional)

If you want to create servable model archives from own checkpoints (e.g., trained on different datasets),
please refer to the archiving scripts (scripts/archive_in_docker.sh).
Change the arguments accordingly in the script before running.
It's mostly bookkeeping by replacing the data name and/or checkpoint paths; the script should be self-explanatory. Then execute the scripts with
```
sh scripts/archive_in_docker.sh
```
The servable model archive (.mar) will be generated under ./mars. Serving newly archived models is straightforward; simply replace the `--models` args in `scripts/serve_{cpu,gpu}_in_{docker,singularity}.sh`

with the new model name and .mar. The `--models` flag for torchserve can also take multiple arguments to serve multiple model archives concurrently.
