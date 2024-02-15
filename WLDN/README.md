## WLDN model for mechansim prediction

## Brief summary
The forward predictor based on an implementation of the Weisfeiler-Lehman network architecture developed by Jin et al. An in-depth discussion of the algorithm can be found in Jin et. al. publication [Predicting Organic Reaction Outcomes with Weisfeiler-Lehman Network](https://github.com/wengong-jin/nips17-rexgen)

## Step 0 - Installation
The forward predictor can be installed from source as follows:

```shell
git clone https://github.com/jfjoung/mechanism_prediction.git
cd mechanism_prediction/WLDN
conda env create --file environment.yml
conda activate wlnfw
pip install .
```
The environment.yml file contains all the essential python dependencies which are needed to train/test/run the forward predictor.
To use the forward predictor with GPUs, the following packages will need to be installed on top of the packages present in the environment_wlnfw.yml environment-file:

```shell
conda install -c anaconda cudatoolkit=10.1 cudnn=8.0
pip install tensorflow-gpu==2.4
```

Note that during installation of these additional packages, the wlnfw environment needs to be activated.

Additional note: The installed versions of Tensorflow, CUDA Toolkit, and Nvidia drivers must all be compatible. Please see [here](https://www.tensorflow.org/install/source#gpu) for compatibility between Tensorflow and CUDA Toolkit, and [here](https://docs.nvidia.com/deploy/cuda-compatibility/) for compatibility between CUDA Toolkit and Nvidia drivers.

## Step 1 - Downloading and preprocessing reaction data

After making the mechanistic dataset using `Mechanistic_dataset_preparation/Mechanistic_dataset_generation.ipynb`, you will get the `.txt.proc` file. 
The reaction SMILES are fully atom-mapped, the reaction edits are specified after SMILES string in `.txt.proc` file. 


## Step 2 - Training a model
Before training can be initiated, a folder containing three .txt.proc files, respectively corresponding to a training, validation and test set, is required.
Model training is handled by the `bin/train_all.py` script:

```shell
python bin/train_all.py --nproc 4 --train data/train.txt.proc --valid data/valid.txt.proc --test data/test.txt.proc --model-name mech_pred --model-dir model
```

The trained models are saved in .hdf5, as well as in .tf, format in the model_dir specified in the the corresponding command line argument (_vide supra_). Summaries of the training history for both the core and ranker model are stored in the same folder in .json format; the hyperparameters settings are stored in -params.txt files.

The individual parts of the forward predictor can be trained separately by specifying the `--part` flag. Option `1` indicates that only the core model needs to be trained and that the final reactive core needs to provided for every data-point in training and validation set. Option `2` indicates that only the ranker model needs to be trained.

The default hyperparameter values ought to already provide acceptable results, but ideally, the various hyperparameters should be optimized for the specific data set under consideration.

A full list of command line arguments can be found below:
  * `--nproc` (int) : Number of Processors to use for generation of input graphs (default=1)
  * `--model-dir` (str) : Directory that the model will be saved in. If using Docker/Singularity be sure this is in the mounted volume (default=os.path.join(os.path.dirname(os.path.realpath(\_\_file\_\_)), 'models'))
  * `--train` (str) : Path to the training data (default='../data/train.txt.proc')
  * `--valid` (str) : Path to the validation data (default='../data/valid.txt.proc')
  * `--test` (str) : Path to the test data (default='../data/test.txt.proc')
  * `--reagents` (bool): To use reagents (i.e., molecules that do not contribute atoms to product) during training (store_true)
  * `--model-name`  (str) : Name of the model (default='wln_fw_pred')
  * `--batch-size` (int) : Size of batches for training the WLN_Core model (default=10)
  * `--hidden-bond-classifier` (int) : Hidden size of the dense layers of the bond classifier model (default=300)
  * `--hidden-cand-ranker` (int) : Hidden size of the dense layers of the candidate ranker model (default=500)
  * `--depth` (int) : Depth of the graph convolutions, similar to fingerprint radius (default=3)
  * `--epochs` (int) : Number of epochs to train. Early stopping is enabled so use the `--early-stopping` flag to modify the patience (default=10)
  * `--learning-rate` (int) : Learning rate (default=0.001)
  * `--early-stopping` (int) : How many epochs to run until there is no more progress in minimizing loss (only the best epoch is saved). If set to 0 the model will train for the number of epochs using flag `--epochs` (default=2)
  * `--clipnorm` (float) : Gradient clipping value (default=5.0)
  * `--cutoff` (float) : Cutoff for accuracy before training second candidate ranker model (default=0.6)
  * `--resume` (int) : Resume on step. 0=train all, 1=start from core model training, 2=start from candidate products generation, 3=start from candidate ranker model' (default=0)

## Step 3 - Testing the model

The performance of the model (in terms of topk-agreement between predicted and recorded SMILES) can be tested with the help of the `bin/test_all.py` script:

```shell
python bin/test_all.py --test data/test.txt.proc --model-name mech_pred --model-dir model
```

In a first step, the testing script predicts the reactive core for each datapoint in the test set and outputs this to a `.cbond_detailed.txt` file in the model directory. Subsequently, the potential products are constructed and ranked; the top ranked candidates are outputted to a .predictions.txt file. Finally, SMILES are constructed for every candidate, and these are compared to the recorded SMILES (candidate SMILES are outputted to a `.predictions.txt.eval_by_smiles` file). Initiating testing at each of these three individually steps can be specified with the help of the `--resume` command line argument.

A full list of command line arguments can be found below:
  * `--model-dir` (str) : Directory that the model will be loaded from (default=os.path.join(os.path.dirname(os.path.realpath(\_\_file\_\_)), 'models'))
  * `--model-name`  (str) : Name of the model (default='wln_fw_pred')
  * `--test` (str) : Path to the test data (default='../data/test.txt.proc')
  * `--reagents` (bool): To use reagents (i.e., molecules that do not contribute atoms to product) during testing (store_true)
  * `--batch-size` (int) : Size of batches for training the WLN_Core model (default=10)
  * `--resume` (int) : Resume on step. 0=test all, 1=start from reactive bond prediction , 2=start from candidate products construction, 3=only evaluate ranked candidates by smiles (default=0)

## Step 4 - Prediction
The reaction mechanism from a set of reactants can be predicted using the `bin/predict.py` script:

```shell
python bin/predict.py --smiles "CCCCCCN.O=C(O)c1cccc(CBr)c1"
```

It will print the possible reaction pathways in the format of,
```
{Accumulate_probability: {'reactions': [a list of chemicals for elementary steps], 'prob': [a list of a probabiltiy for elementary steps], 'predicted_edits': [a list of edits for elementary steps], 'score': [a list of score for elementary steps], 'rank': [a list of rank for each elementary step]}}


{0.8647005051771167: {'reaction': ['CCCCCCN.O=C(O)c1cccc(CBr)c1', 'CCCCCC[NH2+]C([O-])(O)c1cccc(CBr)c1', 'CCCCCCNC([O-])(O)c1cccc(CBr)c1', 'CCCCCCNC(=O)c1cccc(CBr)c1.[OH-]', 'CCCCCCNC(=O)c1cccc(CBr)c1.[OH-]'], 'prob': [1, 0.864800298454965, 1.0, 1.0, 0.9998846], 'predicted_edits': [[], [(6, 8, 1.0, 'b'), (7, 8, 1.0, 'b'), (6, 5, 2, 'f'), (7, 8, 1, 'f')], [(6, 5, 0, 'h'), (6, 5, 1, 'f')], [(7, 9, 0.0, 'b'), (7, 8, 2.0, 'b'), (9, 7, 1, 'f'), (8, 7, 2, 'f')], []], 'score': [0, -18.715117, 25.234385, 216.97578, -25.030212], 'rank': [1, 1, 1, 1, 1]},
0.134605929707336: {'reaction': ['CCCCCCN.O=C(O)c1cccc(CBr)c1', 'CCCCCC[NH2+]Cc1cccc(C(=O)O)c1.[Br-]', 'CCCCCCNCc1cccc(C(=O)O)c1.[Br-]', 'CCCCCCNCc1cccc(C(=O)O)c1.[Br-]'], 'prob': [1, 0.1350925015621326, 0.99999404, 0.9964042], 'predicted_edits': [[], [(15, 16, 0.0, 'b'), (6, 15, 1.0, 'b'), (16, 15, 1, 'f'), (6, 5, 2, 'f')], [(6, 5, 0, 'h'), (6, 5, 1, 'f')], []], 'score': [0, -20.571655, -12.999102, -25.030212], 'rank': [1, 2, 1, 1]},
0.0004752531801971084: {'reaction': ['CCCCCCN.O=C(O)c1cccc(CBr)c1', 'CCCCCC[NH2+]Cc1cccc(C(=O)O)c1.[Br-]', 'CCCCCCNCc1cccc(C(=O)O)c1.[Br-]', 'CCCCCC[NH+]1Cc2cccc(c2)C1([O-])O.[Br-]', 'CCCCCCN1Cc2cccc(c2)C1([O-])O.[Br-]', 'CCCCCCN1Cc2cccc(c2)C1=O.[Br-].[OH-]', 'CCCCCCN1Cc2cccc(c2)C1=O.[Br-].[OH-]'], 'prob': [1, 0.1350925015621326, 0.99999404, 0.0035180103, 0.99999845, 1.0, 0.9999999], 'predicted_edits': [[], [(15, 16, 0.0, 'b'), (6, 15, 1.0, 'b'), (16, 15, 1, 'f'), (6, 5, 2, 'f')], [(6, 5, 0, 'h'), (6, 5, 1, 'f')], [(13, 14, 1.0, 'b'), (6, 13, 1.0, 'b'), (14, 13, 1, 'f'), (6, 5, 2, 'f')], [(6, 5, 0, 'h'), (6, 5, 1, 'f')], [(14, 16, 0.0, 'b'), (14, 15, 2.0, 'b'), (16, 14, 1, 'f'), (15, 14, 2, 'f')], []], 'score': [0, -20.571655, -12.999102, -30.67647, 66.12041, 164.423, -23.111406], 'rank': [1, 2, 1, 2, 1, 1, 1]}}
```

