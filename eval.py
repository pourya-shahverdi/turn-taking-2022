from datetime import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

class MAEMetric(keras.metrics.MeanAbsoluteError):
    def __init__(self, window, name='mean_absolute_error', dtype=None):
        super().__init__(name=f'{name}/{window}', dtype=dtype)
        self.window = window
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true[:, :self.window], y_pred[:, :self.window], sample_weight=sample_weight)

def build_model(lr=0.001, dropout=0.0, regularization=0.001, units=60, num_features=11):
    print(f'Building model with lr {lr}, dropout {dropout}, regularization {regularization}, units {units}')
    model = keras.Sequential()
    model.add(layers.LSTM(units, kernel_regularizer=keras.regularizers.L2(regularization), dropout=dropout, stateful=True, batch_input_shape=[1, 1, num_features]))
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

def compute_deviation(model: keras.Model, frames):
    for i, frame in enumerate(frames):
        print(frame.shape)
        if frame.shape[0] > 120:
            prediction = model.predict(tf.expand_dims(frame[:-80], 0))[0]
            plt.figure()
            plt.hist(np.arange(0, 60), 60, weights=prediction)
            (Path(dir)/'images').mkdir(exist_ok=True)
            plt.savefig(Path(dir)/'images'/f'hist{i}.png')

def save_data(model: keras.Model, frames, path):
    results = []
    for frame in frames:
        spoken = frame[:, -1]
        predictions = []
        model.reset_states()
        for sample in tqdm(frame):
            result = model.predict(tf.expand_dims(tf.expand_dims(sample[:-1], 0), 0), batch_size=1)[0]
            predictions.append(result)
        model.reset_states()
        results.append((spoken, predictions))
    np.save(path, results)
    return results

def compute_pr(results, increment=2):
    precision = []
    recall = []
    for window in range(2, 60, increment):
        tp = 0
        fp = 0
        fn = 0
        for spoken, predictions in results:
            total = np.zeros([len(spoken) + window])
            for j, prediction in enumerate(predictions):
                total += np.pad(prediction[:window], (j, len(spoken)-j))
            total = total[:-window]
            speaking = False
            print(total.shape)
            for j, score in enumerate(total):
                if score > (window / 2) and not speaking:
                    speaking = True
                    lower_bound = max(j-20, 0)
                    upper_bound = min(j+20, len(spoken))
                    if np.max(spoken[lower_bound:upper_bound]) == 1:
                        tp += 1
                    else:
                        fp += 1
                elif score < (window / 2.5):
                    speaking = False
            speaking = False
            for j, label in enumerate(spoken):
                if label == 1 and not speaking:
                    speaking = True
                    lower_bound = max(j-20, 0)
                    upper_bound = min(j+20, len(total))
                    if np.max(total[lower_bound:upper_bound]) < (window / 2):
                        fn += 1
                elif label == 0:
                    speaking = False
        print(f'fp {fp}, tp {tp}, fn {fn}')
        precision.append(tp / (tp + fp))
        recall.append(tp / (tp + fn)) 
            # plt.figure()
            # plt.subplot(3, 1, 1)
            # plt.hist(np.arange(0, len(max_scores)), len(max_scores), weights=max_scores)
            # plt.subplot(3, 1, 2)
            # plt.hist(np.arange(0, len(spoken)), len(spoken), weights=spoken)
            # plt.subplot(3, 1, 3)
            # plt.hist(np.arange(0, len(total)), len(total), weights=total)
            # plt.savefig(Path('/tmp')/'images'/f'hist{i}_{window}.png')
        # set = tf.keras.preprocessing.timeseries_dataset_from_array(frame, None, 120, batch_size=1)

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
    parser.add_argument('--models', nargs='+', default=[])
    parser.add_argument('--name', default=None)
    args = parser.parse_args()
    for model_name in args.models:
        # print(model_name)
        with open(Path(model_name) / 'hparams.txt') as m:
            feature = m.readline().split(':')[-1][1:]
            print(feature)
        evaluation = make_dataframes(['6'], features=feature)
        model = build_model(units=15, num_features=len(evaluation[0][0])-1)
        model.load_weights(Path(model_name) / 'checkpoint')
        results_path: Path = Path('/home/alex/turn-taking/data/results') / args.name / feature / 'data.npy'
        if results_path.exists():
            results = np.load(results_path, allow_pickle=True)
        else:
            results_path.parent.mkdir(exist_ok=True, parents=True)
            results = save_data(model, evaluation, results_path)
        compute_pr(results)
    # plot_histograms(model, evaluation, '/tmp')


if __name__ == '__main__':
    main()