import json

import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import save_model

from WLN.data_loading import Candidate_DataLoader
from WLN.models import WLNCandidateRanker


def wln_diffnet(train=None, valid=None, test=None, batch_size=1, hidden=500,
                    depth=3, epochs=10, learning_rate=0.001, early_stopping=0,
                    clipnorm=5.0, model_name='mech_pred', model_dir='models',
                    use_multiprocessing=True, workers=1):

    """
    Trains the candidate ranker.

    Args:
        train (str): path to training data
        valid (str): path to validation data
        test (str): path to test data
        model_name (str): the name of your model will be saved in models folder
        model_dir (str): path to the folder where the models and intermediate output will be written
        batch_size (int): batch size for training, 1 is currently the only batch_size supported
        hidden (int): the number of hidden layers in the WLN network
        depth (int): the depth the WLN graph convolutions will traverse
        epochs (int): number of epochs, only the best model is saved
        learning_rate (int): the learning rate
        clipnorm (int): gradient clipping value
        early_stopping (int): patience for early stopping
        use_multiprocessing (bool): multiprocessing will be used
        workers (int): number of processors to use by multiprocessing

    Returns:
        model: the trained model, the history can be accessed with model.history
    """

    assert train and valid, 'Please specify a training set and valid set'

    train_detailed = f'{model_dir}/train_{model_name}.cbond_detailed.txt'
    valid_detailed = f'{model_dir}/valid_{model_name}.cbond_detailed.txt'

    train_gen = Candidate_DataLoader(train_detailed, batch_size)
    val_gen = Candidate_DataLoader(valid_detailed, batch_size)

    assert len(train_gen) > 0, f'Training set has {len(train_gen)} examples, has to be greater than 0'
    assert len(val_gen) > 0,   f'Validation set has {len(val_gen)} examples, has to be greater than 0'

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=100000,
        decay_rate=0.9,
        staircase=True)

    model = WLNCandidateRanker(hidden, depth)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=clipnorm)
    model.compile(
        optimizer=opt,
        loss=tf.nn.softmax_cross_entropy_with_logits,
        metrics=[ #TODO add a metric for ranking accuracy?
            tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1_accuracy'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top_10_accuracy'),
            # top_20_acc,
            # top_100_acc,
        ],
    )

    model_name = model_name + '_diffnet'
    model_output = f'{model_dir}/{model_name}-weights.hdf5'
    history_output = f'{model_dir}/{model_name}-history.json'
    params_output = f'{model_dir}/{model_name}-params.txt'
    callbacks = []
    callbacks.append(ModelCheckpoint(model_output, monitor='val_loss', save_best_only=True, save_weights_only=True))

    if early_stopping != 0:
        print(f'EARLY STOPPING USED WITH PATIENCE OF {early_stopping} EPOCHS')
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping,
                restore_best_weights=True
            )
        )

    model_history = model.fit(
        train_gen, epochs=epochs,
        validation_data=val_gen,
        shuffle=False,
        callbacks=callbacks,
        use_multiprocessing=use_multiprocessing,
        workers=workers,
        verbose=1
    )
    with open(params_output, 'w') as f:
        f.write(json.dumps({'batch_size':batch_size,
        'hidden':hidden,
        'depth':depth,
        'epochs':epochs,
        'learning_rate':learning_rate,
        'clipnorm':clipnorm}))

    pd.DataFrame(model_history.history).to_json(history_output)

    save_model(model, f'{model_dir}/{model_name}-full.tf', save_format=tf)

    return model

if __name__ == '__main__':
    wln_diffnet(train='models/train_mech_pred.cbond_detailed.txt', valid='models/valid_mech_pred.cbond_detailed.txt',
                test='models/valid_mech_pred.cbond_detailed.txt', hidden=10, epochs=1, workers=20)
