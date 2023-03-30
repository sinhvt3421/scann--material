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

def process_smfe(data_path='experiments/smfe/', save_path=''):
    
    all_files = sorted(glob.glob(data_path + '/*/*.xyz'))
    print("Loading all files :", len(all_files))
    
    all_struct = []
    idx = 0
    for f in all_files:
        mols = pybel.readfile("xyz", f)
        for mol in mols:
            data = mol.title.split('\"')
            lattice = np.array(data[1].split(),'float32').reshape(3,3)
            
            properties = {'f_en': data[-2]}
            
            atoms = [x.OBAtom for x in mol.atoms]
            coordinates = np.array([x.coords for x in mol.atoms],dtype='float32')

            atomics = [x.GetAtomicNum() for x in atoms]
            atomic_symbols = [atoms_symbol[str(x)] for x in atomics]

            nstruct = {'id': idx, 'Properties': properties,
                    'Atoms': atomic_symbols, 'Atomic': atomics,
                    'Coords': coordinates, 'Lattice':lattice}

            all_struct.append(nstruct)
            idx += 1
            
    print("Saving file and removing temp dirs")

    dataset = 'smfe'
    dataset_file = os.path.join(
        save_path, dataset, dataset + '_data_energy.npy')
    if not os.path.exists(os.path.join(save_path, dataset)):
        os.makedirs(os.path.join(save_path, dataset))

    np.save(dataset_file, all_struct)