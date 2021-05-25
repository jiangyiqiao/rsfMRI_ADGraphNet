import os
import sys
import pickle
import argparse
from numpy.lib.format import open_memmap

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from feeder.feeder_mci import Feeder

toolbar_width = 30


def print_toolbar(rate, annotation=''):
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


def gendata(
        data_path,
        label_path,
        data_out_path,
        label_out_path,
        max_frame=130):
    num_person_out = 1
    feeder = Feeder(
        data_path=data_path,
        label_path=label_path,
        window_size=max_frame)

    sample_nums = feeder.sample_nums
    sample_label = []

    fp = open_memmap(
        data_out_path,
        dtype='float32',
        mode='w+',
        shape=(sample_nums, 90, max_frame, 90, num_person_out))

    for i in range(sample_nums):
        data, label = feeder[i]
        print_toolbar(i * 1.0 / sample_nums,
                      '({:>5}/{:<5}) Processing data: '.format(
                          i + 1, sample_nums))
        fp[i, :, 0:data.shape[1], :, :] = data
        sample_label.append(label)

    with open(label_out_path, 'wb') as f:
        pickle.dump((list(sample_label)), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MCI Data Converter.')
    parser.add_argument(
        '--data_path', default='data/NEL')
    parser.add_argument(
        '--out_folder', default='data/NEL')
    parser.add_argument(
        '--r1', default='0.8')
    arg = parser.parse_args()

    for fold in range(1, 6):
        part = ['train', 'val']
        for p in part:
            data_path = '{}/mci_{}/{}/data_{}.mat'.format(arg.data_path, p, arg.r1, str(fold))
            label_path = '{}/{}/mci_{}_label.json'.format(arg.data_path, arg.r1, p)
            data_out_path = '{}/mci_{}/{}/{}_data_{}.npy'.format(arg.out_folder, p, arg.r1, p, fold)
            label_out_path = '{}/mci_{}/{}/{}_label_{}.pkl'.format(arg.out_folder, p, arg.r1, p, fold)

            if not os.path.exists(arg.out_folder):
                os.makedirs(arg.out_folder)
            gendata(data_path, label_path, data_out_path, label_out_path)
