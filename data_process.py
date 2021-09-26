import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from torch import randperm
from torch.utils.data import Subset
from torch._utils import _accumulate
import os


def process_test(data_path):
    df_left = pd.read_csv(os.path.join(data_path, '_slash_xsens_slash_left_tcp.csv'))
    df_right = pd.read_csv(os.path.join(data_path, '_slash_xsens_slash_right_tcp.csv'))

    print(df_left.head(3))

    ax = plt.axes(projection='3d')

    # 三维散点的数据
    zdata = df_left['z'][:200]
    xdata = df_left['x'][:200]
    ydata = df_left['y'][:200]
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

    zdata = df_right['z'][:200]
    xdata = df_right['x'][:200]
    ydata = df_right['y'][:200]
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Oranges')

    plt.show()


def data_process(data_path):
    # For getting the delta of data
    df_left = pd.read_csv(os.path.join(data_path, '_slash_xsens_slash_left_tcp.csv'))
    df_right = pd.read_csv(os.path.join(data_path, '_slash_xsens_slash_right_tcp.csv'))

    zdata_l = np.array(df_left['z'])
    xdata_l = np.array(df_left['x'])
    ydata_l = np.array(df_left['y'])
    qxdata_l = np.array(df_left['x.1'])
    qydata_l = np.array(df_left['y.1'])
    qzdata_l = np.array(df_left['z.1'])
    qwdata_l = np.array(df_left['w'])

    zdata_r = np.array(df_right['z'])
    xdata_r = np.array(df_right['x'])
    ydata_r = np.array(df_right['y'])
    qxdata_r = np.array(df_right['x.1'])
    qydata_r = np.array(df_right['y.1'])
    qzdata_r = np.array(df_right['z.1'])
    qwdata_r = np.array(df_right['w'])

    def get_delta(arr, arr_name):
        arr_shift = np.zeros_like(arr)
        arr_shift[:-1] = arr[1:]
        delta_arr = arr - arr_shift
        delta_arr += 0.5
        delta_arr *= 10000
        print('{}: {}, {}'.format(arr_name, np.max(delta_arr), np.min(delta_arr)))
        return delta_arr

    delta_zdata_l = get_delta(zdata_l, 'zdata_l')
    delta_xdata_l = get_delta(xdata_l, 'xdata_l')
    delta_ydata_l = get_delta(ydata_l, 'ydata_l')

    delta_zdata_r = get_delta(zdata_r, 'zdata_r')
    delta_xdata_r = get_delta(xdata_r, 'xdata_r')
    delta_ydata_r = get_delta(ydata_r, 'ydata_r')

    delta_pos_l = delta_xdata_l, delta_ydata_l, delta_zdata_l
    delta_pos_r = delta_xdata_r, delta_ydata_r, delta_zdata_r
    rot_l = qxdata_l, qydata_l, qzdata_l, qwdata_l
    rot_r = qxdata_r, qydata_r, qzdata_r, qwdata_r

    return delta_pos_l, delta_pos_r, rot_l, rot_r


def manually_random_split(dataset, lengths, sequence_len, generator):
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths),
                       generator=generator).tolist()  # Randomly shuffle a sequence of numbers less than sum(lengths)
    # train_indices, val_indices = [indices[offset - length: offset] for offset, length in zip(_accumulate(lengths), lengths)]

    return [Subset(dataset, indices[offset - length: offset]) for offset, length in zip(_accumulate(lengths), lengths)]


if __name__ == '__main__':
    # data_process('./data/002-chen-04-dualarmstirfry')
    process_test('./data/002-chen-04-dualarmstirfry')
    process_test('./data/002-chen-02-dualarmstirfry')
