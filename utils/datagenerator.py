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
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .atomic_data import atomic_numbers
from ase.units import Hartree, eV, kcal, mol
RNG_SEED = 123
logger = logging.getLogger(__name__)


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

    def __init__(self, type, batch_size=32, nsample=0,
                 indices=None, data_neighbor=None, data_energy=None,
                 converter=True, use_ring=False,
                 centers=np.linspace(0, 4, 20)):
        """
        Params:
            intensive: Whether divide properties by number of atoms in a structure, default 1
            converter: Whether to convert Ha to meV
            type: train, valid, test
        """
        self.batch_size = batch_size
        self.n_jobs = 2
        self.type = type

        self.n = len(indices)
        self.num_batch = ceil(self.n / self.batch_size)
        self.indices = indices
        self.shuffle = True if (type == 'train') else False

        self.data_neighbor = data_neighbor
        self.engery = data_energy
    
        self.use_ring = use_ring
        self.intensive = True

        if converter:
            self.converter = 1000
        else:
            self.converter = 1.0

        self.expand = GaussianDistance(centers)

    def _get_batches_of_transformed_samples(self, idx) :
        m_len = []
        batch_nei = self.data_neighbor[idx]
        batch_atom = self.engery[idx]

        for p in batch_nei:
            for n in p:
                m_len.append(len(n))

        max_length = max(m_len)
        bs = len(batch_nei)

        at = [p[0] for p in batch_atom]
   
        energy = [float(p[1]) * self.converter for p in batch_atom]

        if self.use_ring:
            extra_info = [np.stack([p[2], p[3]], -1) for p in batch_atom]

        local_neighbor = [[[n[2] for n in lc] for lc in p] for p in batch_nei]
        local_weight = [[[n[1] for n in lc] for lc in p] for p in batch_nei]

        local_distance = [[self.expand.convert([float(n[3]) for n in lc])
                           for lc in p] for p in batch_nei]

        # Padding neighbor for each atoms
        pad_local = [pad_sequences(
            lc, padding='post', maxlen=max_length, value=1000) for lc in local_neighbor]
        pad_local = pad_sequences(pad_local, padding='post', value=1000)

        pad_local = np.array(pad_local, dtype=np.int32)

        mask_local = np.ones_like(pad_local)
        mask_local[pad_local == 1000] = 0
        pad_local[pad_local == 1000] = 0

        # Padding local weight and distance
        pad_local_weight = pad_sequences([pad_sequences(
            lc, padding='post', dtype='float32', maxlen=max_length) for lc in local_weight], padding='post', dtype='float32')
        pad_local_distance = pad_sequences([pad_sequences(
            lc, padding='post', dtype='float32', maxlen=max_length) for lc in local_distance], padding='post', dtype='float32')

        # Padding atomic numbers of atom
        pad_atom = np.array(pad_sequences(
            at, padding='post', value=0), dtype='int32')

        if self.use_ring:
            pad_extra = np.array(pad_sequences(
                extra_info, padding='post', value=0), dtype='int32')

        mask_atom = np.ones_like(pad_atom)
        mask_atom[pad_atom == 0] = 0

        inputs = {'atomic': pad_atom, 'mask_atom': np.expand_dims(mask_atom, -1),
                  'locals': pad_local, 'mask_local': mask_local,
                  'local_weight': np.expand_dims(pad_local_weight, -1),
                  'local_distance': np.array(pad_local_distance)}


        if self.use_ring:
            inputs['ring_aromatic'] = pad_extra

        return (inputs,
                {'gam_net': np.array(energy, dtype='float32')})

    def iterator(self):
        pool = ThreadPoolExecutor(self.n_jobs)

        while 1:
            if self.shuffle:
                shuffle(self.indices)
            current_index = 0  # Run a single I/O thread in paralle

            current_index = self.batch_size

            future = pool.submit(self._get_batches_of_transformed_samples,
                                 self.indices[:self.batch_size])

            current_index = self.batch_size
            for i in range(self.num_batch - 1):
                # wait([future])
                minibatch = future.result()
                # While the current minibatch is being consumed, prepare the next
                future = pool.submit(self._get_batches_of_transformed_samples,
                                     self.indices[current_index: current_index +
                                                  self.batch_size],)

                yield minibatch
                current_index += self.batch_size
            # Wait on the last minibatch
            # wait([future])
            minibatch = future.result()
            yield minibatch