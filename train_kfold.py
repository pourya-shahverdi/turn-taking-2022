from datetime import datetime
from pathlib import Path
from sys import path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import argparse

class MAEMetric(keras.metrics.MeanAbsoluteError):
    def __init__(self, window, name='mean_absolute_error', dtype=None):
        super().__init__(name=f'{name}/{window}', dtype=dtype)
        self.window = window
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true[:, :self.window], y_pred[:, :self.window], sample_weight=sample_weight)


def build_model(lr=0.001, dropout=0.0, regularization=0.001, units=60):
    print(f'Building model with lr {lr}, dropout {dropout}, regularization {regularization}, units {units}')
    model = keras.Sequential()
    model.add(layers.LSTM(units, kernel_regularizer=keras.regularizers.L2(regularization), dropout=dropout))
    model.add(layers.Dense(60, activation='sigmoid', kernel_regularizer=keras.regularizers.L2(regularization)))
    metrics = [MAEMetric(60), MAEMetric(40), MAEMetric(20), MAEMetric(10), MAEMetric(5)]
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=lr), loss=keras.losses.MeanSquaredError(), metrics=metrics)
    return model

def make_dataframes(files, features='all'):
    dataframes = []
    for file in files:
        for run in ['a', 'b', 'c']:
            feature_path = (Path('/home/alex/turn-taking/data/merged_features/') / f'AB_{file}_{run}').with_suffix('.csv')
            data = np.genfromtxt(feature_path, delimiter=',')[:, 1:-2]
            speakers = data[:, 8:10]
            prosody = data[:, 0:3]
            print(f'Prosody shape {prosody.shape}')
            prosody = np.where(speakers[:, 0:1] == 0, prosody, 0)
            print(f'Prosody shape {prosody.shape}')
            gesture = data[:, 3:8]
            asr = data[:, 10:]
            if features == 'all':
                out_data = np.concatenate([prosody, gesture, asr, speakers], -1)
            elif features == 'nogesture':
                out_data = np.concatenate([prosody, asr, speakers], -1)
            elif features == 'gesture':
                out_data = np.concatenate([gesture, speakers], -1)
            elif features == 'prosody':
                out_data = np.concatenate([prosody, speakers], -1)
            elif features == 'asr':
                out_data = np.concatenate([asr, speakers], -1)
            elif features == 'silence':
                out_data = speakers
            # print(data.shape)
            speaking = False
            start_index = 0
            for i, line in enumerate(speakers):
                if not speaking and line[0] == 1:
                    speaking = True
                    start_index = i
                    frame = out_data[start_index:]
                    dataframes.append(frame)
                    break
                elif speaking and line[1] == 1:
                    speaking = False
    return dataframes

def make_dataset(files, features='all'):
    dataframes = make_dataframes(files, features=features)
                    # print(frame.shape)
    datasets = []
    for frame in dataframes:
        datasets.append(tf.keras.preprocessing.timeseries_dataset_from_array(frame, None, 200, batch_size=1, sequence_stride=2))
        print(f'Dataset length {len(datasets[-1])}')
    dataset = tf.data.experimental.sample_from_datasets(datasets)
    def separate_labels(x):
        interviewee = x[0, :-60]
        interviewer = x[0, -60:]
        label = interviewer[:, -1]
        return interviewee[:, :-1], label
    return dataset.map(separate_labels)

def save_error_plot(history, dir):
    mae_3s = history['val_mean_absolute_error/60']
    mae_2s = history['val_mean_absolute_error/40']
    mae_1s = history['val_mean_absolute_error/20']
    mae_half_s = history['val_mean_absolute_error/10']
    mae_quarter_s = history['val_mean_absolute_error/5']
    plt.figure()
    plt.plot(mae_3s, label='3s')
    plt.plot(mae_2s, label='2s')
    plt.plot(mae_1s, label='1s')
    plt.plot(mae_half_s, label='500ms')
    plt.plot(mae_quarter_s, label='250ms')
    plt.xlabel("Training Epochs")
    plt.ylabel("Mean Absolute Error")
    plt.ylim(bottom=0)
    plt.grid(axis='y')
    plt.legend()
    (Path(dir)/'images').mkdir()
    plt.savefig(Path(dir)/'images'/'error_plot.png')
    with open(Path(dir)/'evaluation.txt', 'w') as f:
        f.write(f'{mae_quarter_s[-1]},{mae_half_s[-1]},{mae_1s[-1]},{mae_2s[-1]},{mae_3s[-1]}')

def plot_histograms(model: keras.Model, frames, dir):
    for i, frame in enumerate(frames):
        print(frame.shape)
        if frame.shape[0] > 120:
            prediction = model.predict(tf.expand_dims(frame[:-90, :-1], 0))[0]
            print(prediction.shape)
            plt.figure()
            plt.hist(np.arange(0, 60), 60, weights=prediction)
            (Path(dir)/'images').mkdir(exist_ok=True)
            plt.savefig(Path(dir)/'images'/f'hist{i}.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.00025, type=float)
    parser.add_argument('--batch', default=4, type=int)
    parser.add_argument('--regularization', default=0.002, type=float)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--units', default=60, type=int)
    parser.add_argument('--features', default='all', type=str)
    parser.add_argument('--dir', default='k-fold', type=str)
    args = parser.parse_args()
    print(f'Building dataset with {args.features} features')
    samples = ['1', '2', '4', '5', '6']
    for i in range(len(samples)):
        train_samples = samples[:i] + samples[i+1:]
        test_samples = [samples[i]]
        # print(train_samples)
        ds = make_dataset(train_samples, features=args.features)
        val_ds = make_dataset(test_samples, features=args.features).batch(640)
        batched = ds.shuffle(10000).batch(args.batch)
        print(batched.element_spec)
        logdir = Path("logs/") / args.dir / f'k-{i}' / args.features
        logdir.mkdir(exist_ok=True, parents=True)
        with open(Path(logdir)/'hparams.txt', 'w') as f:
            f.write(f"lr: {args.lr}, dropout: {args.dropout}, reg: {args.regularization}, units: {args.units}, batch: {args.batch}, features: {args.features}")
        model = build_model(args.lr, args.dropout, args.regularization, args.units)
        history = model.fit(batched, epochs=60, validation_data=val_ds, callbacks=[keras.callbacks.TensorBoard(log_dir=logdir)])
        save_error_plot(history.history, logdir)
        model.save(Path(logdir) / 'checkpoint')


if __name__ == '__main__':
    main()