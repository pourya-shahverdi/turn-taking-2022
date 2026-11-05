import csv
from pathlib import Path
from math import isclose

def main():
    files = ['1', '2', '4', '5', '6']
    for file in files:
        for run in ['a', 'b', 'c']:
            text_path = (Path('/home/alex/turn-taking/data/scored_text/') / f'AB_{file}_{run}').with_suffix('.csv')
            prosody_path = (Path('/home/alex/turn-taking/data/prosody/') / f'B_{file}_{run}').with_suffix('.csv')
            gesture_path = (Path('/home/alex/turn-taking/data/reduced_pose/') / f'B_{file}_{run}').with_suffix('.csv')
            out_path = (Path('/home/alex/turn-taking/data/merged_features/') / f'AB_{file}_{run}').with_suffix('.csv')
            with open(text_path) as t, open(prosody_path) as p, open(gesture_path) as g, open(out_path, 'w') as o:
                text = csv.reader(t)
                prosody = csv.reader(p)
                gesture = csv.reader(g)
                out = csv.writer(o)
                next_t = next(text, None)
                next_p = next(prosody, None)
                next_g = next(gesture, None)
                current_time = float(next_p[0])
                # sp1, sp2, confidence, score, txt, txt
                text_features = [0, 0, 0, 0, "", ""]
                while(next_p):
                    if next_t:
                        if current_time > float(next_t[1]):
                            # Save current word to state
                            if next_t[3]:
                                text_features[2] = next_t[2]
                                text_features[3] = next_t[5]
                            text_features[4] = next_t[3]
                            text_features[5] = next_t[4]
                            next_t = next(text, None)
                        if next_t:
                            if current_time < float(next_t[0]):
                                text_features[0] = 0
                                text_features[1] = 0
                            elif current_time > float(next_t[0]) and current_time < float(next_t[1]):
                                text_features[0] = 1 if next_t[3] else 0
                                text_features[1] = 1 if next_t[4] else 0
                        else:
                            text_features[0] = 0
                            text_features[1] = 0
                    else:
                        text_features[0] = 0
                        text_features[1] = 0
                    if isclose(current_time, float(next_g[0]), rel_tol=0, abs_tol=0.01):
                        gesture_features = next_g[1:]
                        out_row = next_p + gesture_features + text_features
                        out.writerow(out_row)
                        next_p = next(prosody, None)
                        next_g = next(gesture, None)
                        current_time += 0.05
                    else:
                        current_time += 0.05
                        next_p = next(prosody, None)
                     
                    
                

if __name__ == '__main__':
    main()