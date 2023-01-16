import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor, wait, as_completed
from math import ceil
import numpy as np
from ase.db import connect
import numpy as np
from random import shuffle
import tensorflow as tf
from ase.units import Hartree, eV, kcal, mol
from tensorflow.keras.preprocessing.sequence import pad_sequences

RNG_SEED = 123
logger = logging.getLogger(__name__)


def pad_nested_sequences(sequences, max_len_1, max_len_2, dtype='int32', value=0):
    """
        Pad 3D array 
    Args:
        sequences (list): 3D array axis=(0,1,2)
        max_len_1 (int): Max length inside axis = 2
        max_len_2 (int): Max length outside axis = 1
        dtype (str, optional):  Defaults to 'int32'.
        value (int, optional): Padding value. Defaults to 0.

    Returns:
        np.ndarray: Padded sequences
    """
    pad_sq = [pad_sequences(
        sq, padding='post', maxlen=max_len_1, value=value) for sq in sequences]
    pad_sq = pad_sequences(pad_sq, padding='post',
                           maxlen=max_len_2, value=value)
    pad_sq = np.array(pad_sq, dtype=dtype)
    return pad_sq


class GaussianDistance():
    """
    Expand distance with Gaussian basis sit at centers and with width 0.5.
    """

    def __init__(self, centers: np.ndarray = np.linspace(0, 4, 20), width=0.5):
        """
        Args:
            centers: (np.array) centers for the Gaussian basis
            width: (float) width of Gaussian basis
        """
        self.centers = centers
        self.width = width

    def convert(self, d: np.ndarray):
        """
        expand distance vector d with given parameters
        Args:
            d: (1d array) distance array
        Returns
            (matrix) N*M matrix with N the length of d and M the length of centers
        """
        d = np.array(d)
        return np.exp(-((d[:, None] - self.centers[None, :]) ** 2) / self.width ** 2)


class DataIterator(object):
    """
    Create Data interator over dataset
    """

    def __init__(self, data_energy, data_neighbor, indices, batch_size=32,
                 converter=True, use_ring=False,
                 centers=np.linspace(0, 4, 20)):
        """
        Args:
            data_energy (list): _description_
            data_neighbor (list): _description_
            indices (dict): Index for train, valid, test
            batch_size (int, optional): Defaults to 32.
            converter (bool, optional): Scale target value. Defaults to True.
            use_ring (bool, optional): Use ring aromatic information. Defaults to False.
            centers (list, optional): Gaussian expand centers. Defaults to np.linspace(0, 4, 20).
        """

        self.batch_size = batch_size
        self.n_jobs = 2

        self.indices = indices
        self.num_batch = {}
        self.get_num_batch()

        self.data_neighbor = data_neighbor
        self.data_energy = data_energy

        self.use_ring = use_ring
        self.intensive = True

        if converter:
            self.converter = 1000
        else:
            self.converter = 1.0

        self.expand = GaussianDistance(centers)

    def get_num_batch(self):
        for type in self.indices:
            self.num_batch[type] = ceil(
                len(self.indices[type])/self.batch_size)

    def get_batch(self, idx):
        """
            data_neighbor: [(site_x_label, nn['site_index'], w, d),...]
            data_energy:   ['Atomic', 'Properties', 'Ring_info']
        """
        neighbor_len = []
        center_len = []

        batch_nei = self.data_neighbor[idx]
        batch_atom = self.data_energy[idx]

        for c in batch_nei:
            center_len.append(len(c))
            for n in c:
                neighbor_len.append(len(n))

        max_length_neighbor = max(neighbor_len)
        max_length_center = max(center_len)

        energy = np.array(
            [float(p[1]) * self.converter for p in batch_atom], 'float32')

        # Padding neighbor for each atoms
        local_neighbor = [[[n[1] for n in lc] for lc in p] for p in batch_nei]
        pad_local = pad_nested_sequences(
            local_neighbor, max_length_neighbor, max_length_center, value=1000, dtype='int32')
        mask_local = (pad_local != 1000)
        pad_local[pad_local == 1000] = 0

        # Padding local weight and distance
        local_weight = [[[n[2] for n in lc] for lc in p] for p in batch_nei]
        local_distance = [[self.expand.convert([float(n[3]) for n in lc])
                           for lc in p] for p in batch_nei]
        pad_local_weight = pad_nested_sequences(
            local_weight, max_length_neighbor, max_length_center, dtype='float32')

        pad_local_distance = pad_nested_sequences(
            local_distance, max_length_neighbor, max_length_center, dtype='float32')

        # Padding atomic numbers of atom
        atomics = [center[0] for center in batch_atom]
        pad_atom = np.array(pad_sequences(
            atomics, padding='post', value=0), dtype='int32')

        mask_atom = (pad_atom != 0)

        if self.use_ring:
            extra_info = [np.stack([center[2], center[3]], -1)
                          for center in batch_atom]
            pad_extra = np.array(pad_sequences(
                extra_info, padding='post', value=0), dtype='int32')

        inputs = {'atomic': pad_atom, 'atom_mask': np.expand_dims(mask_atom, -1),
                  'neighbors': pad_local, 'neighbor_mask': mask_local,
                  'neighbor_weight': np.expand_dims(pad_local_weight, -1),
                  'neighbor_distance': pad_local_distance}

        if self.use_ring:
            inputs['ring_aromatic'] = pad_extra

        return (inputs, energy)

    def iterator(self, type):

        pool = ThreadPoolExecutor(self.n_jobs)

        while 1:
            if type == 'train':
                shuffle(self.indices[type])
            current_index = 0  # Run a single I/O thread in paralle

            current_index = self.batch_size

            future = pool.submit(self.get_batch,
                                 self.indices[type][:self.batch_size])

            current_index = self.batch_size
            for i in range(self.num_batch[type] - 1):
                # wait([future])
                minibatch = future.result()
                # While the current minibatch is being consumed, prepare the next
                future = pool.submit(self.get_batch,
                                     self.indices[type][current_index: current_index +
                                                        self.batch_size],)

                yield minibatch
                current_index += self.batch_size
            # Wait on the last minibatch
            # wait([future])
            minibatch = future.result()
            yield minibatch
