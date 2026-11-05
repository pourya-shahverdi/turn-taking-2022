from rosbag.bag import Bag
import math
from pathlib import Path
import csv
from rospy import Time
from tf_bag import BagTfTransformer
import tf2_py
import rospy

def main():
    bag_dir = Path('/media/alex/easystore/Interview/')
    out_dir = Path('/home/alex/turn-taking/data/raw_pose')
    for interviewee_bag_f in (bag_dir / 'Interviewee-08-16-2021').glob('**/*.bag'):
        print(f'Reading bag {interviewee_bag_f}')
        tf_buffer = tf2_py.BufferCore(rospy.Duration(1000000))
        interviewee_bag = Bag(interviewee_bag_f)
        start_time = math.ceil(interviewee_bag.get_start_time() * 20) / 20
        end_time = math.floor(interviewee_bag.get_end_time() * 20) / 20
        current_time = start_time 
        outfile = (out_dir / interviewee_bag_f.stem).with_suffix('.csv')
        print(start_time)
        print(end_time)
        for topic, msg, t in interviewee_bag.read_messages(topics=['tf', 'tf_static']):
            for msg_tf in msg.transforms:
                if topic == '/tf_static':
                    tf_buffer.set_transform_static(msg_tf, "default_authority")
                else:
                    tf_buffer.set_transform(msg_tf, "default_authority")
        with open(outfile, 'w') as f:
            writer = csv.writer(f)
            while current_time < end_time:
                try:
                    lh = tf_buffer.lookup_transform_core('torso', 'left_hand', Time.from_sec(current_time)).transform.translation
                    rh = tf_buffer.lookup_transform_core('torso', 'right_hand', Time.from_sec(current_time)).transform.translation
                    le = tf_buffer.lookup_transform_core('torso', 'left_elbow', Time.from_sec(current_time)).transform.translation
                    re = tf_buffer.lookup_transform_core('torso', 'right_elbow', Time.from_sec(current_time)).transform.translation
                    he = tf_buffer.lookup_transform_core('torso', 'head', Time.from_sec(current_time)).transform.translation
                    outrow = [current_time]
                    outrow += [lh.x, lh.y, lh.z]
                    outrow += [rh.x, rh.y, rh.z]
                    outrow += [le.x, le.y, le.z]
                    outrow += [re.x, re.y, re.z]
                    outrow += [he.x, he.y, he.z]
                    writer.writerow(outrow)
                except Exception as e:
                    print(e)
                    print(f'Skipping time {current_time} as no data is available')
                current_time += 0.05

if __name__ == '__main__':
    main()