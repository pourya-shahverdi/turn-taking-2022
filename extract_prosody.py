import json
from librosa import feature
import yaml
from rosbag.bag import Bag
from pathlib import Path
import csv
import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import math
from sklearn.preprocessing import StandardScaler

def main():
    wav_dir = Path('/media/alex/easystore/Interview/Interviewee-08-16-2021/')
    out_dir = Path('/home/alex/turn-taking/data/prosody')
    for wav_file in wav_dir.glob('**/*.wav'):
        bagname = wav_file.stem
        print(bagname)
        bagfile = (wav_dir / bagname).with_suffix('.bag')
        info_dict = yaml.safe_load(Bag(bagfile, 'r')._get_yaml_info())
        bag_time = math.ceil(info_dict['start'] * 20) / 20
        # print(info_dict)
        raw_data, rate = librosa.load(wav_file, sr=16000)
        # print(raw_data.shape)
        # print(librosa.get_duration(raw_data, sr=16000))
        pitches, magnitudes = librosa.piptrack(raw_data, sr=16000, hop_length=800)
        pitches = np.transpose(pitches)
        max_pitch = np.argmax(magnitudes, axis=0)
        pitch = np.take_along_axis(pitches, np.reshape(max_pitch, [-1, 1]), axis=1)
        energy = librosa.feature.rms(raw_data, hop_length=800)
        energy = librosa.amplitude_to_db(energy)
        energy = np.reshape(energy, [-1])
        onset_env = librosa.onset.onset_strength(y=raw_data, sr=16000, hop_length=800)
        data = np.stack([max_pitch, energy, onset_env], -1)
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        outfile = (out_dir / bagname).with_suffix('.csv')
        with open(outfile, 'w') as f:
            writer = csv.writer(f)
            current_time = bag_time
            for row in data:
                writer.writerow([current_time, row[0], row[1], row[2]])
                current_time += 0.05

if __name__ == '__main__':
    main()