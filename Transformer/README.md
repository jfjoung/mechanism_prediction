# Augmented Transformer

Benchmarking and serving modules for reaction outcome prediction with Augmented Transformer, reimplemented using OpenNMT based on the manuscript (https://www.nature.com/articles/s41467-020-19266-y).

## Benchmarking (GPU Required)

### Step 1/4: Environment Setup
```
bash scripts/setup_gpu.sh
```

### Step 2/4: Data Preparation

Prepare the raw .csv files for train, validation and test (atom mapping not required for all). The required columns are "id" and "rxn_smiles", where "rxn_smiles" are NON atom-mapped reaction SMILES.

### Step 3/4: Path Configuration

Configure the environment variables in ./scripts/benchmark.sh, especially the paths, to point to the *absolute* paths of raw files and desired output paths.
```
export DATA_NAME="mech"
export TRAIN_FILE=$PWD/data/mech/raw/raw_train.csv
export VAL_FILE=$PWD/data/mech/raw/raw_val.csv
export TEST_FILE=$PWD/data/mech/raw/raw_test.csv
...
```

### Step 4/4: Benchmarking

- Run benchmarking on a machine with GPU using
```
bash scripts/benchmark.sh
```
This will run the preprocessing, training and predicting (uncomment the scripts as necessary) for Augmented Transformer with Top-n accuracies up to n=20 as the final outputs. Progress and result logs will be saved under ./logs

### Try out pretrained models
Download the model from the link [https://www.dropbox.com/scl/fo/o6siqbp1s2offg641al94/h?rlkey=46p2rvhldmobq2z25oi5di4ho&dl=0]
- Open new folder `Transformer/checkpoints/mech/unseen_reactions` and load the unseen test set csv into the directory
- Open new folder `Transformer/checkpoints/mech` and load the downloaded models into the directory
- Configure model unseen test set csv filename under `benchmark.sh`
```
...
export TEST_UNSEEN_PATH="$PWD/data/mech/unseen_reactions/test.csv"
...
```
- Configure model checkpoint file name under `beam_search.sh`
```
export CHECKPOINT="model_step_1250000.pt"
```

- Run beam search on held out test set reactants using by uncommenting `bash scripts/beam_search.sh` and run
```
bash scripts/benchmark.sh
```

