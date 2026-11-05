from datetime import datetime
from pathlib import Path
import re
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
            print(f"Making dataframes with features{features}")
            feature_path = (Path('/home/alex/turn-taking/data/merged_features/') / f'AB_{file}_{run}').with_suffix('.csv')
            data = np.genfromtxt(feature_path, delimiter=',')[:, 1:-2]
            speakers = data[:, 8:10]
            prosody = data[:, 0:3]
            prosody = np.where(speakers[:, 0:1] == 0, prosody, 0)
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

def compute_pr(results, window=10, increment=5, deviation=20):
    precision = []
    recall = []
    tps = []
    fps = []
    fns = []
    for threshold in range(20, 100, increment):
        threshold = (threshold / 100) * window
        tp = 0
        fp = 0
        fn = 0
        for spoken, predictions in results:
            total = np.zeros([len(spoken) + window])
            for j, prediction in enumerate(predictions):
                total += np.pad(prediction[:window], (j, len(spoken)-j))
            total = total[:-window]
            speaking = False
            # print(total.shape)
            for j, score in enumerate(total):
                if score > threshold and not speaking:
                    speaking = True
                    lower_bound = max(j-deviation, 0)
                    upper_bound = min(j+deviation, len(spoken))
                    if np.max(spoken[lower_bound:upper_bound]) == 1:
                        pass
                    else:
                        fp += 1
                elif score < threshold:
                    speaking = False
            speaking = False
            for j, label in enumerate(spoken):
                if label == 1 and not speaking:
                    speaking = True
                    lower_bound = max(j-deviation, 0)
                    upper_bound = min(j+deviation, len(total))
                    if np.max(total[lower_bound:upper_bound]) < threshold:
                        fn += 1
                    else:
                        tp += 1
                elif label == 0:
                    speaking = False
        # print(f'fp {fp}, tp {tp}, fn {fn}')
        # precision.append(fn)
        tps.append(tp)
        fns.append(fn)
        fps.append(fp)
        if tp > 0 or fp > 0:
            precision.append(tp / (tp + fp))
        else:
            precision.append(0)
        if tp > 0 or fn > 0:
            recall.append(tp / (tp + fn))
        else:
            recall.append(0)

    return precision, recall, [thresh/100 for thresh in range(20, 100, increment)]


def plot_cumulative(results, window=40, name=''):
    for i, (spoken, predictions) in enumerate(results):
        total = np.zeros([len(spoken) + window])
        for j, prediction in enumerate(predictions):
            total += np.pad(prediction[:window], (j, len(spoken)-j))
        total = total[:-window]
        total = total / window        
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.ylabel('Speaker Label')
        plt.hist(np.arange(0, len(spoken)), len(spoken), weights=spoken)
        plt.subplot(2, 1, 2)
        plt.xlabel('Frames')
        plt.ylabel('Accumulated Predictions')
        plt.hist(np.arange(0, len(total)), len(total), weights=total)
        plt.savefig(Path('/tmp')/'images'/f'hist{i}_{name}.png')

def compute_f1(precision, recall):
    f1 = []
    for p, r in zip(precision, recall):
        if p + r > 0:
            f1.append((2 * p * r)/(p + r))
        else:
            f1.append(0)
    return f1

def compute_deviation(results, window=20, deviation=40):
    early_rate = []
    late_rate = []
    # for threshold in range(20, 100, increment):
    #     threshold = (threshold / 100) * window
    #     tp = 0
    #     fp = 0
    #     fn = 0
    for threshold in tqdm(range(0, 100, 1)):
        threshold = (threshold / 100) * window
        speaking = False
        early = 0
        late = 0
        for spoken, predictions in results:
            total = np.zeros([len(spoken) + window])
            for j, prediction in enumerate(predictions):
                total += np.pad(prediction[:window], (j, len(spoken)-j))
            total = total[:-window]
            for j, score in enumerate(total):
                if score > threshold and not speaking:
                    speaking = True
                    lower_bound = max(j-deviation, 0)
                    upper_bound = min(j+deviation, len(spoken))
                    if np.max(spoken[lower_bound:upper_bound]) == 1:
                        turn = np.argmax(spoken[lower_bound:upper_bound])
                        if turn < j:
                            late += 1
                        else:
                            early += 1
                    else:
                        pass # False Positive
                elif score < threshold:
                    speaking = False
        if late > 0 or early > 0:
            early_rate.append(early / (early + late))
            late_rate.append(late / (late + early))
        else:
            early_rate.append(0)
            late_rate.append(1)
    return early_rate, late_rate, [thresh/100 for thresh in range(0, 100, 1)]


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
    parser.add_argument('models')
    parser.add_argument('--name', default='k-fold')
    args = parser.parse_args()
    root_path = Path(args.models)
    samples = ['1', '2', '4', '5', '6']
    accumulated_results = {}
    accumulated_scores = {}
    for k_fold in root_path.iterdir():
        for model_name in k_fold.iterdir():
            feature = model_name.name
            i = int(k_fold.name[-1])
            test_set = samples[i]
            dataframes = make_dataframes(test_set, features=feature)
            results_path: Path = Path('/home/alex/turn-taking/data/results') / args.name / feature / k_fold.name / 'data.npy'
            if results_path.exists():
                results = np.load(results_path, allow_pickle=True)
            else:
                model = build_model(units=15, num_features=len(dataframes[0][0])-1)
                model.load_weights(model_name / 'checkpoint')
                results_path.parent.mkdir(exist_ok=True, parents=True)
                results = save_data(model, dataframes, results_path)
            if feature in accumulated_results:
                accumulated_results[feature] = np.concatenate([accumulated_results[feature], results], axis=0)
            else:
                accumulated_results[feature] = results

            with open(Path(model_name) / 'evaluation.txt') as m:
                scores = [float(score) for score in m.readline().split(',')]
                if feature in accumulated_scores:
                    accumulated_scores[feature].append(np.array(scores))
                else:
                    accumulated_scores[feature] = [np.array(scores)]

    for feature, scores in accumulated_scores.items():
        mean = np.mean(scores, axis=0)
        print(f'Mean {feature} scores: {mean}')

    labels = {
        'gesture': 'Gesture Cues',
        'prosody': 'Prosodic Cues',
        'silence': 'Silence Cues',
        'asr': 'Verbal Cues',
        'all': 'All Cues',
        'nogesture': 'Verbal+Prosodic Cues'
    }

    # for feature, results in accumulated_results.items():
    #     plt.figure()
    #     for window in tqdm([5, 10, 20, 40, 60]):
    #         precision, recall, threshold = compute_pr(results, window=window, increment=2, deviation=40)
    #         f1 = compute_f1(precision, recall)
    #         plt.plot(threshold, f1, label=str(window))
    #     plt.xlabel("Probability Threshold")
    #     plt.ylabel("F1 Score")
    #     plt.ylim(0, 1)
    #     plt.xlim(0, 1)
    #     plt.legend()
    #     imagedir: Path = Path('/home/alex/turn-taking/data/results') / args.name / feature / 'images'
    #     imagedir.mkdir(exist_ok=True)
    #     plt.savefig(imagedir/'pr.png')
    plt.figure()
    for feature, results in accumulated_results.items():
        precision, recall, threshold = compute_pr(results, window=10, increment=2, deviation=40)
        f1 = compute_f1(precision, recall)
        plt.plot(recall, precision, label=labels[feature])
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.legend()
    imagedir: Path = Path('/home/alex/turn-taking/data/results') / args.name / 'images'
    imagedir.mkdir(exist_ok=True)
    plt.savefig(imagedir/'pr.png')

    # early, late, thresh = compute_deviation(accumulated_results['nogesture'], window=40)
    # plt.figure()
    # plt.plot(thresh, early, label='early')
    # plt.plot(thresh, late, label='late')
    # plt.xlabel("Threshold")
    # plt.ylabel("Early/Late Rate")
    # plt.legend()
    # imagedir: Path = Path('/home/alex/turn-taking/data/results') / args.name / 'images'
    # imagedir.mkdir(exist_ok=True)
    # plt.savefig(imagedir/'deviation-2s.png')
    # plot_cumulative(accumulated_results['nogesture'])

if __name__ == '__main__':
    main()