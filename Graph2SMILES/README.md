# Graph2SMILES

Benchmarking and serving modules for reaction outcome prediction with Graph2SMILES, based on the manuscript (https://pubs.acs.org/doi/abs/10.1021/acs.jcim.2c00321).

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
This will run the preprocessing, training and predicting (uncomment the scripts as necessary) for Graph2SMILES with Top-n accuracies up to n=20 as the final outputs. Progress and result logs will be saved under ./logs

### Try out pretrained models
Download the model and vocab from the link [https://www.dropbox.com/scl/fo/9qagy7mg00jpo1uqr7pr3/h?rlkey=neyjoefseoltukjqgc4cdedjy&dl=0]

- Open new folder `Graph2SMILES/checkpoints/mech/unseen_reactions` and load the unseen test set csv into the directory
- Open new folder `Graph2SMILES/checkpoints/mech` and load the downloaded models into the directory
- Open new folder `Graph2SMILES/data/mech/processed` and load the vocab.txt into the directory

- Configure model unseen test set csv filename under `benchmark.sh`
```
...
export TEST_UNSEEN_PATH="$PWD/data/mech/unseen_reactions/test.csv"
...
```

- Configure model checkpoint file name under `beam_search.sh`
```
export CHECKPOINT="model.850000_84.pt"
```

- Run beam search on held out test set reactants using by uncommenting `bash scripts/beam_search.sh` and run
```
bash scripts/benchmark.sh
```

