import logging
import os
import shutil
import zipfile
import tempfile
import json
from urllib import request as request

import numpy as np
from pymatgen.core import Structure
from .atomic_data import atoms_symbol

logging.getLogger("").setLevel(logging.CRITICAL)
logging.disable(logging.CRITICAL)


def process_mp2018(save_path):
    tmpdir = tempfile.mkdtemp("mp2018")
    print("Download Material Project 2018 data to temp folder ", tmpdir)

    zip_path = os.path.join(tmpdir, "mp.2018.6.1.zip")
    url = "https://ndownloader.figshare.com/files/15087992"

    request.urlretrieve(url, zip_path)
    print("Done.")

    print("Loading files...")
    data = json.loads(zipfile.ZipFile(zip_path).read("mp.2018.6.1.json"))
    print("Done.")

    print("Parse json files...")

    all_struct = []
    for idx, d in enumerate(data):
        if idx % 10000 == 0:
            print("Parse {:6d} / {:6d} MP data".format(idx + 1, len(data)))

        mol = Structure.from_str(d["structure"], fmt="cif")  # Assuming there's only one structure in the CIF

        if len(mol) > 1:
            properties = {"e_f": d["formation_energy_per_atom"], "e_b": d["band_gap"]}

            coordinates = mol.frac_coords
            lattice = mol.lattice.matrix

            atomics = list(mol.atomic_numbers)
            atomic_symbols = [atoms_symbol[str(x)] for x in atomics]

            nstruct = {
                "id": d["material_id"],
                "Properties": properties,
                "Atoms": atomic_symbols,
                "Atomic": atomics,
                "Coords": coordinates,
                "Lattice": lattice,
                "Cartesian": False,
            }
            all_struct.append(nstruct)

    print("Saving file and removing temp dirs")

    dataset = "mp2018"

    dataset_file = os.path.join(save_path, dataset, dataset + "_data_energy.npy")
    if not os.path.exists(os.path.join(save_path, dataset)):
        os.makedirs(os.path.join(save_path, dataset))

    sorted_struct = sorted(all_struct, key=lambda x: len(x["Atoms"]))
    np.save(dataset_file, sorted_struct)

    shutil.rmtree(tmpdir, ignore_errors=True)

    print("Finished clearing temp dirs")
