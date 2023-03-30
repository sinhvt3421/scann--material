import logging
import os
import re
import numpy as np
import shutil

import glob
from urllib import request as request
from openbabel import pybel

from .atomic_data import atoms_symbol, atoms_symbol

logging.getLogger("").setLevel(logging.CRITICAL)
logging.disable(logging.CRITICAL)

def process_fullerence(data_path='experiments/fullerence/', save_path=''):
    
    all_files = sorted(glob.glob(data_path + '*.xyz'))
    print("Loading all files :", all_files)
    
    all_struct = []
    idx = 0
    for f in all_files:
        mols = pybel.readfile("xyz", f)
        for mol in mols:
            prop = mol.title.split()
            properties = {'homo':prop[0],'lumo':prop[1],'total_energy':prop[2]}
            
            atoms = [x.OBAtom for x in mol.atoms]
            coordinates = np.array([x.coords for x in mol.atoms],dtype='float32')

            atomics = [x.GetAtomicNum() for x in atoms]
            atomic_symbols = [atoms_symbol[str(x)] for x in atomics]

            ring_info = [1 if at.IsInRing() else 0 for at in atoms]
            aromatic = [1 if at.IsAromatic() else 0 for at in atoms]

            nstruct = {'id': idx, 'Properties': properties,
                    'Atoms': atomic_symbols, 'Atomic': atomics,
                    'Coords': coordinates, 'Ring': ring_info,
                    'Aromatic': aromatic,}

            all_struct.append(nstruct)
            idx += 1
            
    print("Saving file and removing temp dirs")

    dataset = 'fullerence'
    dataset_file = os.path.join(
        save_path, dataset, dataset + '_data_energy.npy')
    if not os.path.exists(os.path.join(save_path, dataset)):
        os.makedirs(os.path.join(save_path, dataset))

    np.save(dataset_file, all_struct)