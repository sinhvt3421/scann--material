import glob
import logging
import os
import shutil
import zipfile
import tempfile
from urllib import request as request

import numpy as np
from openbabel import pybel

from .atomic_data import atoms_symbol

logging.getLogger("").setLevel(logging.CRITICAL)
logging.disable(logging.CRITICAL)


def process_fullerene(save_path):
    tmpdir = tempfile.mkdtemp("fullerene")

    print("Download Fullerene data to temp folder ", tmpdir)

    zip_path = os.path.join(tmpdir, "fullerene.zip")
    url = "https://zenodo.org/record/8435481/files/fullerene.zip?download=1"

    request.urlretrieve(url, zip_path)
    print("Done.")

    print("Loading files...")

    zipfile.ZipFile(zip_path).extractall(tmpdir)
    all_files = sorted(glob.glob(tmpdir + "/*/*.xyz"))

    print("Loading all files :", all_files)

    all_struct = []
    idx = 0
    for f in all_files:
        mols = pybel.readfile("xyz", f)
        for mol in mols:
            prop = mol.title.split()
            properties = {"homo": prop[0], "lumo": prop[1], "total_energy": prop[2]}

            atoms = [x.OBAtom for x in mol.atoms]
            coordinates = np.array([x.coords for x in mol.atoms], dtype="float32")

            atomics = [x.GetAtomicNum() for x in atoms]
            atomic_symbols = [atoms_symbol[str(x)] for x in atomics]

            ring_info = [1 if at.IsInRing() else 0 for at in atoms]
            aromatic = [1 if at.IsAromatic() else 0 for at in atoms]

            nstruct = {
                "id": idx,
                "Properties": properties,
                "Atoms": atomic_symbols,
                "Atomic": atomics,
                "Coords": coordinates,
                "Features": {
                    "Ring": ring_info,
                    "Aromatic": aromatic,
                },
                "Cartesian": True,
            }

            all_struct.append(nstruct)
            idx += 1

    print("Saving file and removing temp dirs")

    dataset = "fullerene"
    dataset_file = os.path.join(save_path, dataset, dataset + "_data_energy.npy")
    if not os.path.exists(os.path.join(save_path, dataset)):
        os.makedirs(os.path.join(save_path, dataset))

    np.save(dataset_file, all_struct)

    shutil.rmtree(tmpdir, ignore_errors=True)

    print("Finished clearing temp dirs")
