import csv
from pathlib import Path


def main():
    files = ['1', '2', '4', '5', '6']
    for file in files:
        for run in ['a', 'b', 'c']:
            a_path = (Path('/home/alex/turn-taking/data/processed_text/') / f'A_{file}_{run}').with_suffix('.csv')
            b_path = (Path('/home/alex/turn-taking/data/processed_text/') / f'B_{file}_{run}').with_suffix('.csv')
            out_path = (Path('/home/alex/turn-taking/data/merged_text/') / f'AB_{file}_{run}').with_suffix('.csv')
            with open(a_path) as a, open(b_path) as b, open(out_path, 'x') as out:
                a_data = csv.reader(a)
                b_data = csv.reader(b)
                out_data = csv.writer(out)
                a_row = next(a_data)
                b_row = next(b_data)
                while(a_row or b_row):
                    if a_row and (not b_row or float(a_row[1]) < float(b_row[1])):
                        if a_row[4] != '':
                            out_data.writerow(a_row)
                        try:
                            a_row = next(a_data)
                        except StopIteration:
                            a_row = None
                    else:
                        if b_row[3] != '':
                            out_data.writerow(b_row)
                        try:
                            b_row = next(b_data)
                        except StopIteration:
                            b_row = None

if __name__ == '__main__':
    main()
