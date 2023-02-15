import os
import numpy as np
from sklearn.model_selection import train_test_split


def split_data(len_data, test_percent=0.1, train_size=None, test_size=None):
    """
        Split data training
    Args:
        len_data (int): Length of dataset.
        test_percent (float, optional): Percentage of data for testing. Defaults to 0.1.
        train_size (int, optional): If provided, fix the number of train and test size
        test_size (int, optional): If provided, fix the number of train and test size
    Returns:
        _type_: _description_
    """
    if train_size:
        N_train = train_size
        N_test = test_size
    else:
        N_train = int(len_data * (1-test_percent*2))
        N_test = int(len_data * test_percent)

    N_val = len_data - N_train - N_test

    data_perm = np.random.permutation(len_data)
    train, valid, test, extra = np.split(
        data_perm, [N_train, N_train+N_val, N_train + N_val + N_test])
    return train, valid, test, extra


def load_dataset(dataset, dataset_neighbor, target_prop, use_ref=False, use_ring=True, use_hyp=False):
    """
        Load dataset and neighbor information
    Args:
        dataset (str): Path to dataset
        dataset_neighbor (str): Path to dataset neighbor
        target_prop (str): Target property for training
        use_ref (bool, optional): Use reference energy. Defaults to False.
        use_ring (bool, optional): Use ring aromatic information. Defaults to True.

    Returns:
        data_energy (list)
        data_neighbor (list)
    """

    data_full = np.load(dataset, allow_pickle=True)

    data_energy = []
    if use_ref:
        print('Using reference energy optimization')

    if use_ring:
        print('Using ring aromatic information')

    for i, d in enumerate(data_full):
        if use_ring:
            if use_hyp:
                data_energy.append([d['Atomic'], d['Properties'][target_prop],
                                    d['Ring'], d['Aromatic'], d['Acceptor'], d['Donor']])
            else:
                data_energy.append([d['Atomic'], d['Properties'][target_prop],
                                    d['Ring'], d['Aromatic']])
        else:
            if use_ref:
                data_energy.append([d['Atomic'],
                                    d['Properties'][target_prop]-d['Properties']['Ref_energy']])
            else:
                data_energy.append([d['Atomic'], d['Properties'][target_prop]])

    data_energy = np.array(data_energy, dtype='object')

    data_neighbor = np.load(dataset_neighbor, allow_pickle=True)
    data_neighbor = np.array(data_neighbor, dtype='object')

    return data_energy, data_neighbor
