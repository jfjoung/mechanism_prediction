import json
import os

import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import save_model

from WLN.data_loading import Graph_DataLoader
from WLN.metrics import wln_loss, top_10_acc, top_20_acc, top_100_acc
from WLN.models import WLNPairwiseAtomClassifier
from graph_utils.ioutils_direct import nbos


def train_wln_core(train=None, valid=None, reagents=False, batch_size=10, hidden=300,
                    depth=3, epochs=10, output_dim=nbos, learning_rate=0.001,
                    clipnorm=5.0, model_name='wln_core', model_dir='models',
                    early_stopping=0, use_multiprocessing=True, workers=1):

    """
    Trains the first of two networks for the template free forward predictor.

    Args:
        train (str): path to training data
        valid (str): path to validation data
        reagents (bool): use reagents during training
        model_name (str): the name of your model will be saved in models folder
        batch_size (int): batch size for training, depends on your GPU memory how large batches can be
        model_dir (str): path to the folder where the models and intermediate output will be written
        hidden (int): the number of hidden layers in the WLN network
        depth (int): the depth the WLN graph convolutions will traverse
        epochs (int): number of epochs, only the best model is saved
        output_dim (int): the output dimension, should be equal to number of bond classes (see ioutils)
        learning_rate (int): the learning rate
        clipnorm (int): gradient clipping value
        early_stopping (int): patience for early stopping
        use_multiprocessing (bool): multiprocessing will be used
        workers (int): number of processors to use by multiprocessing

    Returns:
        model: the trained model, the history can be accessed with model.history

    """

    assert train and valid, 'Please specify a training set and valid set'

    if reagents:
        print("USING REAGENTS DURING TRAINING")

    train_gen = Graph_DataLoader(train, batch_size, reagents=reagents)
    val_gen = Graph_DataLoader(valid, batch_size, reagents=reagents)

    assert len(train_gen) > 0, f'Training set has {len(train_gen)} examples, has to be greater than 0'
    assert len(val_gen) > 0,   f'Validation set has {len(val_gen)} examples, has to be greater than 0'

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=20000,
        decay_rate=0.9,
        staircase=True)

    model = WLNPairwiseAtomClassifier(hidden, depth, output_dim=output_dim) #hardcoded output_dim to number of different bonds
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=clipnorm)
    model.compile(
        optimizer=opt,
        loss=wln_loss(batch_size),
        metrics=[
            top_10_acc,
            top_20_acc,
            top_100_acc,
        ],
    )

    if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    model_name = model_name + '_core'
    model_output = f'{model_dir}/{model_name}-weights.hdf5'
    history_output = f'{model_dir}/{model_name}-history.json'
    params_output = f'{model_dir}/{model_name}-params.txt'

    callbacks = []
    if early_stopping != 0:
        print(f'EARLY STOPPING USED WITH PATIENCE OF {early_stopping} EPOCHS')
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping,
                restore_best_weights=True
            )
        )

    callbacks.append(ModelCheckpoint(model_output,
                    monitor='val_loss', save_weights_only=True, save_best_only=True))

    model_history = model.fit(
        train_gen, epochs=epochs,
        validation_data=val_gen,
        shuffle=False,
        callbacks=callbacks,
        use_multiprocessing=use_multiprocessing,
        workers=workers
    )

    pd.DataFrame(model_history.history).to_json(history_output)

    #Write params so model can be loaded by the candidate generator
    with open(params_output, 'w') as f:
        f.write(json.dumps({'batch_size':batch_size,
        'hidden':hidden,
        'depth':depth,
        'epochs':epochs,
        'output_dim':output_dim,
        'learning_rate':learning_rate,
        'clipnorm':clipnorm}))

    save_model(model, f'{model_dir}/{model_name}-full.tf', save_format=tf)

    return model

if __name__ == '__main__':
    # model = train_wln_core(train='/work/data/train_trunc.txt.proc', valid='/work/data/valid_trunc.txt.proc',
    #                 hidden=100, epochs=2, workers=4)
    # model = train_wln_core(train='data/elementary_Pistachio_database8.txt.proc', valid='data/elementary_Pistachio_database8.txt.proc',
    #                        hidden=5, epochs=1, workers=4, model_name='testing_8')
    model = train_wln_core(train='data/training_set.txt.proc', valid='data/validation_set.txt.proc',
                          hidden=20, epochs=1, workers=10, model_name='testing')
