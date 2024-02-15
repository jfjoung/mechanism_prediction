## Incorporating a new forward predictor model into ASKCOS

This guide will walk you through (re)training a forward predictor model and incorporating the output into a deployed ASKCOS instance. This walkthrough is designed to be performed with ASKCOS version 2022.01.

The code repository for training the model is available at
https://gitlab.com/mlpds_mit/ASKCOS/wln-keras-fw/.

## Brief summary

The forward predictor used in ASKCOS consists of two distinct models which are interconnected. First, there is the core predictor, which is a graph-neural network, assigning scores to each potential bond edit, and selecting a reactive core of most likely active bonds. Subsequently, potential products are constructed by taking combinations of the bonds in this reactive core. These potential products are finally passed through a second graph-neural network, which ranks the products according to their likelihood to be formed.

At the core of both the core predictor and the candidate ranker is an implementation of the Weisfeiler-Lehman network architecture developed by Jin et al. An in-depth discussion of the algorithm can be found in Jin et. al. publication [Predicting Organic Reaction Outcomes with Weisfeiler-Lehman Network](http://papers.nips.cc/paper/6854-predicting-organic-reaction-outcomes-with-weisfeiler-lehman-network.pdf).

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

The USPTO data can be downloaded with the help of the script `data_prep/download_uspto.py`. The downloaded data is distributed over three distinct files, corresponding to the common train/validation/test-split used for this dataset. Once untarred, the resulting .txt.proc files already have the correct format to be introduced as input to the forward predictor, and hence, no additional preprocessing steps are required.

In case one starts from a new data set which has not yet been preprocessed, a series of steps need to be taken. First, starting from a .txt file consisting of raw/unprocessed reaction SMILES, the `data_prep/preprocess_rxn_smiles.py` needs to be executed:

```shell
python data_prep/preprocess_rxn_smiles.py --data-file 'data/rxn_smiles.txt' --output-file 'data/curated_rxn_smiles.txt'
```

In this script, all reagents (e.g., catalyst or solvent) are first moved to the reactant side of the reaction SMILES. Subsequently, the SMILES are checked for the presence of unmapped atoms and if so, a renumbering, based on an atom-map dictionary and the atom-indices on the reactant side, is attempted. If the latter fails, the reaction SMILES are remapped with the help of the IBM (attention-based) RXNMapper. In order for this script to work with the RXNMapper, the corresponding package needs to be installed and a compatible CONDA environment needs to be activated (cf. the [RXNMapper repository](https://github.com/rxn4chemistry/rxnmapper) for detailed installation instructions). Alternatively, other atom-mapping software can be inserted into the script instead.

Once the reaction SMILES are fully atom-mapped, the reaction core/bond edits needs to be determined for each of them and reactions which inherently cannot be treated by the forward predictor need to be filtered out. This can be done with the help of the `data_prep/prep_data.py` script:

```shell
python data_prep/prep_data.py --data-file 'data/curated_rxn_smiles.txt' [--keep-invalid]
```

This script takes as input the path to a .txt file (specified through the `--data-file` command line argument) containing the (fully) atom-mapped reaction SMILES, and outputs a .txt.proc file with the same lay-out as the downloadable uspto-files (_vide supra_). Upon finishing this preparation step, the script provides statistics about the number of invalid reactions that inherently cannot be recovered by the forward predictor. By default, these invalid reactions are filtered out; they can be retained by specifying the `--keep-invalid` command line argument.

The .txt.proc file can be split (randomly) in train/validation/test sets with the help of the script `data_prep/split_data.py` as follows:

```shell
python data_prep/split_data.py --proc-file 'data/curated_rxn_smiles.txt.proc' --output-base-name 'dataset' --test-set-split 10
```

The output of this script consists of 3 files, {output_base_name}_train.txt.proc, {output_base_name}_valid.txt.proc and {output_base_name}_test.txt.proc, in the same location as the input .txt.proc-file. The `--test-set-split` command line argument indicates the percentage of the dataset that is sampled for the test set (the same value is taken for the splitting of the train and validation set).

Finally, it should also be noted that since the current versions of the models cannot treat chirality, all chiral tags need to be removed from the reaction SMILES before executing the data preparation script(s) in order to facilitate comparison between predicted and recorded product SMILES (cf. step 3).

## Step 2 - Training a model

Before training can be initiated, a folder containing three .txt.proc files, respectively corresponding to a training, validation and test set, is required (cf. step 1).

Model training is handled by the `bin/train_all.py` script:

```shell
python bin/train_all.py --nproc 4 --train data/train.txt.proc --valid data/valid.txt.proc --test data/test.txt.proc --model-name uspto_500k --model-dir uspto_500k
```

The trained models are saved in .hdf5, as well as in .tf, format in the model_dir specified in the the corresponding command line argument (_vide supra_). Summaries of the training history for both the core and ranker model are stored in the same folder in .json format; the hyperparameters settings are stored in -params.txt files.

By default, the model filters out candidate bonds involving reagents during training. Reagent bonds can be included by adding the `--reagents` command line option:

```shell
python bin/train_all.py --nproc 4 --train data/train.txt.proc --valid data/valid.txt.proc --test data/test.txt.proc --model-name uspto_500k_reagents --model-dir uspto_500k_reagents --reagents
```

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
python bin/test_all.py --test data/test.txt.proc --model-name uspto_500k --model-dir uspto_500k
```

To evaluate the performance of the forward predictor model trained with reagent bonds included in the reactive core, the `--reagents` argument has to be included:

```shell
python bin/test_all.py --test data/test.txt.proc --model-name uspto_500k --model-dir uspto_500k_reagents --reagents
```

In a first step, the testing script predicts the reactive core for each datapoint in the test set and outputs this to a `.cbond_detailed.txt` file in the model directory. Subsequently, the potential products are constructed and ranked; the top ranked candidates are outputted to a .predictions.txt file. Finally, SMILES are constructed for every candidate, and these are compared to the recorded SMILES (candidate SMILES are outputted to a `.predictions.txt.eval_by_smiles` file). Initiating testing at each of these three individually steps can be specified with the help of the `--resume` command line argument.

A full list of command line arguments can be found below:
  * `--model-dir` (str) : Directory that the model will be loaded from (default=os.path.join(os.path.dirname(os.path.realpath(\_\_file\_\_)), 'models'))
  * `--model-name`  (str) : Name of the model (default='wln_fw_pred')
  * `--test` (str) : Path to the test data (default='../data/test.txt.proc')
  * `--reagents` (bool): To use reagents (i.e., molecules that do not contribute atoms to product) during testing (store_true)
  * `--batch-size` (int) : Size of batches for training the WLN_Core model (default=10)
  * `--resume` (int) : Resume on step. 0=test all, 1=start from reactive bond prediction , 2=start from candidate products construction, 3=only evaluate ranked candidates by smiles (default=0)

## Step 4 - Preparing Tensorflow serving image for ASKCOS

Once you have the trained model, you will need to create a Tensorflow serving Docker image to load the model into ASKCOS.

The first step in doing so is to convert the models to a format used by Tensorflow serving. A utility script `convert_model.py` is provided for this purpose.

```shell
python bin/convert_model.py --model-name=uspto_500k --model-dir=uspto_500k
```

If you changed any model parameters, you should also pass the modified parameters to this script. The full list of command line arguments can be found below:

  * `--model-dir` (str) : Directory to load the trained model from (default='bin/models'))
  * `--model-name`  (str) : Name of the model (default='wln_fw_pred')
  * `--output-dir`  (str) : Directory to save the converted model files to (default='bin/models')
  * `--hidden-bond-classifier` (int) : Hidden size of the dense layers of the bond classifier model (default=300)
  * `--hidden-cand-ranker` (int) : Hidden size of the dense layers of the candidate ranker model (default=500)
  * `--depth` (int) : Depth of the graph convolutions, similar to fingerprint radius (default=3)
  * `--cutoff` (float) : Cutoff for accuracy before training second candidate ranker model (default=0.6)
  * `--reagents` (bool): To use reagents (i.e., molecules that do not contribute atoms to product) during training (store_true)

Once you have the converted model files, you can build the Docker image using the Dockerfile located in the `tfx` directory.

```shell
docker build -f tfx/Dockerfile --build-arg MODEL_DIR='uspto_500k' --build-arg MODEL_NAME='uspto_500k' -t tfx-wln-fw-uspto:1.0 .
```

The above command will create a Docker image tagged `tfx-wln-fw-uspto:1.0`. The Docker image tag can be anything you like, as long as the same tag is used when updating the ASKCOS deployment in Step 5 below.

## Step 5 - Updating ASKCOS deployment

Once the Tensorflow serving Docker image is built, the final step is to update the ASKCOS deployment. The remainder of the tutorial will involve the askcos-deploy repository, which you should have already used to deploy ASKCOS initially. If not, please see the ASKCOS deployment instructions for the latest release, 2022.01.

### Docker Compose

If you're using Docker Compose deployment, the first file to update is `docker-compose.yml`, by adding a new service block for the new Tensorflow serving image:

```yaml
  tfx-wln-fw-uspto:
    image: tfx-wln-fw-uspto:1.0
    restart: always
    expose:
      - 8501
```

Then, you need to update `model_config.yaml`, which is used by ASKCOS to determine the models available:

```yaml
- type: wln_forward  # used by ASKCOS, do not change
  training_set: uspto_500k_custom  # will appear in model selection menus, should be unique (default model in ASKCOS is uspto_500k)
  name: wln_forward_uspto_500k  # for display only
  version: 1  # model version from tensorflow serving configuration
  description: Template-free forward predictor using WLN architecture.  # for display only
  attributes:
    core_model_name: core  # model name from tensorflow serving models.config
    rank_model_name: rank  # model name from tensorflow serving models.config
    cutoff: 1500  # model parameter from training
    core_size: 16  # model parameter from training
    reagents: true  # model parameter from training
  connection:
    framework: tensorflow  # used by ASKCOS, do not change
    hostname: tfx-wln-fw-uspto  # should match service name from docker-compose.yml
    port: 8501  # do not change
```

You may need to restart the deployment for the change to `model_config.yaml` to take effect:

```shell
docker-compose down && bash deploy.sh start
```

Finally, you will need to start the new Tensorflow serving service:

```shell
docker-compose up -d tfx-wln-fw-uspto
```

Once all that is done, you should be able to see the new model under the server status page, and the new training set name as an option on the forward prediction page.

### Kubernetes

For Kubernetes deployment, the Helm values file needs to be updated with the new service. Under the `mlserver` block, a new list entry should be added:

```yaml
  - name: tfx-wln-fw-uspto
    image:
      repository: tfx-wln-fw-uspto
      pullPolicy: IfNotPresent
      tag: "1.0"
    replicaCount: 1
    service:
      type: ClusterIP
      port: 8501
    resources:
      limits:
        memory: "12G"
      requests:
        memory: "10G"
    framework: tensorflow
    models:
      - type: wln_forward  # used by ASKCOS, do not change
        training_set: uspto_500k_custom  # will appear in model selection menus, should be unique (default model in ASKCOS is uspto_500k)
        name: wln_forward_uspto_500k  # for display only
        version: 1  # model version from tensorflow serving configuration
        description: Template-free forward predictor using WLN architecture.  # for display only
        attributes:
          core_model_name: core  # model name from tensorflow serving models.config
          rank_model_name: rank  # model name from tensorflow serving models.config
          cutoff: 1500  # model parameter from training
          core_size: 16  # model parameter from training
          reagents: true  # model parameter from training
```

For the changes to take effect, you will either need to reinstall or upgrade the Helm install. Please note that to upgrade the Helm install, you must have the password and erlangCookie for the rabbitmq subchart. See [here](https://github.com/bitnami/charts/tree/master/bitnami/rabbitmq#upgrading) for more information.
