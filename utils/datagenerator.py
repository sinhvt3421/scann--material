import tensorflow as tf
from tensorflow.keras.utils import Sequence
from math import ceil
import numpy as np

SEED=2134
np.random.seed(SEED)

def pad_sequence(sequences, maxlen=None, dtype='int32', value=0, padding='post'):

    num_samples = len(sequences)
    sample_shape = ()

    if maxlen is None:
        lengths = []
        for x in sequences:
            lengths.append(len(x))
        maxlen = np.max(lengths)

    sample_shape = np.asarray(sequences[0]).shape[1:]

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)

    for idx, s in enumerate(sequences):
        trunc = s[-maxlen:]
        trunc = np.asarray(trunc, dtype=dtype)
        x[idx, : len(trunc)] = trunc
    return x


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
    pad_sq = [pad_sequence(
        sq, padding='post', maxlen=max_len_1, value=value,dtype=dtype) for sq in sequences]
    pad_sq = pad_sequence(pad_sq, padding='post',
                          maxlen=max_len_2, value=value,dtype=dtype)
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


class DataIterator(Sequence):
    """
    Create Data interator over dataset
    """

    def __init__(self, data_energy, data_neighbor, batch_size=32,
                 converter=True, use_ring=False,
                 centers=np.linspace(0, 4, 20), shuffle=False):
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
        
        self.shuffle = shuffle
        
        self.data_neighbor = data_neighbor
        self.data_energy = data_energy

        self.use_ring = use_ring
        self.intensive = True

        if converter:
            self.converter = 1000
        else:
            self.converter = 1.0

        self.expand = GaussianDistance(centers)
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data_energy))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __len__(self):
        return ceil(len(self.data_energy) / self.batch_size)
    
    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        
        batch_nei = self.data_neighbor[indexes]
        batch_atom = self.data_energy[indexes]

        neighbor_len = []
        center_len = []
        
        for c in batch_nei:
            center_len.append(len(c))
            for n in c:
                neighbor_len.append(len(n))

        max_length_center = max(center_len)
        max_length_neighbor = max(neighbor_len)

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
        pad_atom = pad_sequence(atomics, padding='post',
                                maxlen=max_length_center, value=0, dtype='int32')

        mask_atom = (pad_atom != 0)

        if self.use_ring:
            extra_info = [np.stack([center[2], center[3]], -1)
                          for center in batch_atom]
            pad_extra = pad_sequence(
                extra_info, padding='post',maxlen=max_length_center, value=0, dtype='int32')

        inputs = {'atomic': pad_atom, 'atom_mask': np.expand_dims(mask_atom, -1),
                  'neighbors': pad_local, 'neighbor_mask': mask_local,
                  'neighbor_weight': np.expand_dims(pad_local_weight, -1),
                  'neighbor_distance': pad_local_distance}

        if self.use_ring:
            inputs['ring_aromatic'] = pad_extra

        return (inputs, energy)
    
            