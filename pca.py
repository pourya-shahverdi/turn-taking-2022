import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

def main():
    in_dir = Path('/home/alex/turn-taking/data/raw_pose')
    out_dir = Path('/home/alex/turn-taking/data/reduced_pose')
    dataframes = []
    paths = []
    for run in in_dir.glob('**/*.csv'):
        data = np.genfromtxt(run, delimiter=',')
        dataframes.append(data)
        paths.append(run)
    combined = np.concatenate(dataframes, 0)[:, 1:]
    print(np.amin(combined))
    print(np.amax(combined))
    scaler = StandardScaler()
    normalized = scaler.fit_transform(combined)
    pca = PCA(n_components=5)
    pca.fit(normalized)
    print(pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_ratio_))
    for i in range(len(dataframes)):
        frame = dataframes[i]
        path = paths[i]
        data = np.concatenate([frame[:, 0:1], pca.transform(scaler.transform(frame[:, 1:]))], -1)
        np.savetxt(out_dir / path.name, data, delimiter=",")
    

if __name__ == '__main__':
    main()