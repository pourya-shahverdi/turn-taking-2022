import json
import yaml
from rosbag.bag import Bag
import os
from pathlib import Path
import csv

def main():
    transcript_dir = '/home/alex/turn-taking/data/transcripts/'
    for transcript in Path(transcript_dir).glob('**/*'):
        csv_path = (transcript.parent.parent / 'processed_text' / transcript.stem.split('.')[0]).with_suffix('.csv')
        print(csv_path)
        bagname = transcript.stem.split('.')[0]
        print(bagname)
        if bagname[0] == 'A':
            bagfile = (Path('/media/alex/easystore/Interview/Interviewer-08-16-2021/') / bagname).with_suffix('.bag')
        else:
            bagfile = (Path('/media/alex/easystore/Interview/Interviewee-08-16-2021/') / bagname).with_suffix('.bag')
        info_dict = yaml.safe_load(Bag(bagfile, 'r')._get_yaml_info())
        bag_time = round(info_dict['start'], 2)
        # print(info_dict)
        with open(transcript) as t, open(csv_path, 'x') as c:
            parsed = json.load(t)
            output = csv.writer(c)
            words = parsed["results"][-1]['alternatives'][0]['words']
            for word in words:
                print(word)
                start_time = bag_time + float(word['startTime'][:-1])
                end_time = bag_time + float(word['endTime'][:-1])
                confidence = word['confidence']
                if word['speakerTag'] == 1:
                    output.writerow([start_time, end_time, confidence, word['word'], ''])
                else:
                    output.writerow([start_time, end_time, confidence, '', word['word']])

if __name__ == '__main__':
    main()