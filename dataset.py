import os
import json
import numpy as np
from torch.utils.data import Dataset

class Feeder_kinetics(Dataset):
    def __init__(self, data_path, label_path, ignore_empty_sample=True, random_choose=False, random_shift=False,
                 random_move=False, window_size=-1, pose_matching=False, num_person_out=2, debug=False):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.pose_matching = pose_matching
        self.ignore_empty_sample = ignore_empty_sample
        self.num_person_out = num_person_out

        # 这里直接定义了C, T, V, M的值，请根据你的实际数据调整
        self.C = 6  # 假设你的数据有6个通道（例如x坐标、y坐标、x速度、y速度、x加速度、y加速度）
        self.T = 205  # 假设你的数据时间长度为205
        self.V = 18  # 假设你的数据有14个节点
        self.M = 1  # 人数由num_person_out参数确定

        self.load_data()

    def load_data(self):
        # Load the list of data samples
        self.sample_name = os.listdir(self.data_path)
        if self.debug:
            self.sample_name = self.sample_name[:100]

        # Load the labels
        with open(self.label_path) as f:
            label_info = json.load(f)
        self.label = {id: label_info[id]['label_index'] for id in label_info}

    def __len__(self):
        return len(self.sample_name)

    def __getitem__(self, index):
        # 加载和处理数据的逻辑
        sample_name = self.sample_name[index]
        sample_path = os.path.join(self.data_path, sample_name)
        with open(sample_path, 'r') as f:
            video_info = json.load(f)

        # 初始化用于存储数据的 numpy 数组
        data_numpy = np.zeros((self.C, self.T, self.V, self.M), dtype=np.float32)

        # 处理关键点
        for frame_index, frame_features in enumerate(video_info):
            if frame_index < self.T:
                coordinates = [feat['coordinate'] for feat in frame_features['features']]
                speeds = [feat['speed'] for feat in frame_features['features']]
                accelerations = [feat.get('acceleration', [0.0, 0.0]) for feat in frame_features['features']]
                if len(coordinates) == self.V and len(speeds) == self.V and len(accelerations) == self.V:
                    coordinates = np.array(coordinates).T  # 转置以匹配 (2, V)
                    speeds = np.array(speeds).T  # 转置以匹配 (2, V)
                    accelerations = np.array(accelerations).T  # 转置以匹配 (2, V)
                    data_numpy[0:2, frame_index, :, 0] = coordinates  # 前两个通道用于存储坐标
                    data_numpy[2:4, frame_index, :, 0] = speeds  # 中间两个通道用于存储速度
                    data_numpy[4:6, frame_index, :, 0] = accelerations  # 后两个通道用于存储加速度
                else:
                    print(f"Frame {frame_index} in file {sample_name} skipped due to shape mismatch: "
                          f"coordinates={len(coordinates)}, speeds={len(speeds)}, accelerations={len(accelerations)}")
                    continue  # 跳过不符合形状要求的帧

        label_id = os.path.splitext(sample_name)[0]
        label = self.label.get(label_id, 0)  # 如果找不到标签，默认值为 0

        # 如果需要，应用数据增强
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label

def gendata(data_path, label_path, data_out_path, label_out_path, num_person_in=1, num_person_out=1, max_frame=205):
    feeder = Feeder_kinetics(
        data_path=data_path,
        label_path=label_path,
        window_size=max_frame,
        num_person_out=num_person_out)

    sample_name = feeder.sample_name
    sample_label = []
    fp = open_memmap(
        data_out_path,
        dtype='float32',
        mode='w+',
        shape=(len(sample_name), 6, max_frame, 18, num_person_out))  # 更新形状以匹配6个通道

    for i, s in enumerate(sample_name):
        try:
            data, label = feeder[i]
        except ValueError as e:
            print(f"Skipping sample {s} due to error: {e}")
            continue
        print_toolbar(i * 1.0 / len(sample_name),
                      '({:>18}/{:<18}) Processing data: '.format(i + 1, len(sample_name)))
        fp[i, :, 0:data.shape[1], :, :] = data
        sample_label.append(label)

    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

if __name__ == '__main__':
    import sys
    import argparse
    import pickle
    from numpy.lib.format import open_memmap

    def print_toolbar(rate, annotation=''):
        toolbar_width = 30
        # setup toolbar
        sys.stdout.write("{}[".format(annotation))
        for i in range(toolbar_width):
            if i * 1.0 / toolbar_width > rate:
                sys.stdout.write(' ')
            else:
                sys.stdout.write('-')
            sys.stdout.flush()
        sys.stdout.write(']\r')

    def end_toolbar():
        sys.stdout.write("\n")


    parser = argparse.ArgumentParser(description='Kinetics-skeleton Data Converter.')

    parser.add_argument('--data_path', default=r'D:\infant program\16', help='Input data path')
    parser.add_argument('--out_folder', default=r'D:\infant program\16', help='Output folder path')
    arg = parser.parse_args()

    part = ['train', 'val']
    for p in part:
        data_path = f'{arg.data_path}/kinetics_{p}'
        label_path = f'{arg.data_path}/kinetics_{p}_label.json'
        data_out_path = f'{arg.out_folder}/{p}_data.npy'
        label_out_path = f'{arg.out_folder}/{p}_label.pkl'

        if not os.path.exists(arg.out_folder):
            os.makedirs(arg.out_folder)
        gendata(data_path, label_path, data_out_path, label_out_path)
