import logging
import os
import shutil
import zipfile
import tempfile
import json
from urllib import request as request

import numpy as np
from ase.units import Ang, Bohr, Debye, Hartree, eV
from openbabel import pybel

logging.getLogger("").setLevel(logging.CRITICAL)
logging.disable(logging.CRITICAL)


atomic_atom = {"1": "H", "6": "C", "7": "N", "8": "O", "9": "F"}
atom_atomic = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}


def process_qm9_std_jctc(save_path):
    tmpdir = tempfile.mkdtemp("qm9")
    print("Download qm9_std_jctc data to temp folder ", tmpdir)

    zip_path = os.path.join(tmpdir, "qm9_std_jctc.zip")
    url = "https://ndownloader.figshare.com/files/28715319"

    request.urlretrieve(url, zip_path)
    print("Done.")

    print("Loading files...")
    data = json.loads(zipfile.ZipFile(zip_path).read("qm9_std_jctc.json"))
    print("Done.")

    print("Parse json files...")

    all_struct = []
    for idx in range(len(data)):
        if idx % 10000 == 0:
            print("Parse {:6d} / 130829 QM9 data".format(idx + 1))

        molecule_data = data[idx]
        xyz_lines = []
        xyz_lines.append("Generated from qm9_std_jctc")
        coords = np.dot(molecule_data["atoms"]["coords"], molecule_data["atoms"]["lattice_mat"])
        for element, coord in zip(molecule_data["atoms"]["elements"], coords):
            xyz_lines.append(f"{element} {coord[0]:.12f} {coord[1]:.12f} {coord[2]:.12f}")
        xyz_content = "\n".join(xyz_lines)

        # Save the XYZ content to a temporary file
        xyz_filename = "temp_molecule.xyz"
        with open(xyz_filename, "w") as xyz_file:
            xyz_file.write(f"{len(molecule_data['atoms']['elements'])}\n")
            xyz_file.write(xyz_content)

        # Read the XYZ file using Pybel
        mol = next(pybel.readfile("xyz", xyz_filename))
        atoms = [x.OBAtom for x in mol.atoms]

        atomics = [x.GetAtomicNum() for x in atoms]
        atomic_symbols = [atomic_atom[str(x)] for x in atomics]

        ring_info = [at.IsInRing() for at in atoms]
        aromatic = [at.IsAromatic() for at in atoms]

        properties = {}
        properties["mu"] = molecule_data["mu"]
        properties["alpha"] = molecule_data["alpha"]
        properties["homo"] = molecule_data["HOMO"]
        properties["lumo"] = molecule_data["LUMO"]
        properties["gap"] = molecule_data["gap"]
        properties["r2"] = molecule_data["R2"]
        properties["zpve"] = molecule_data["ZPVE"]
        properties["U0"] = molecule_data["U0"]
        properties["U"] = molecule_data["U"]
        properties["H"] = molecule_data["H"]
        properties["Cv"] = molecule_data["Cv"]
        properties["G"] = molecule_data["G"]
        properties["omega1"] = molecule_data["omega1"]

        nstruct = {
            "id": molecule_data["id"],
            "Properties": properties,
            "Atoms": atomic_symbols,
            "Atomic": atomics,
            "Coords": coords,
            "Cartesian": True,
            "Features": {
                "Ring": ring_info,
                "Aromatic": aromatic,
            },
        }
        all_struct.append(nstruct)

    print("Saving file and removing temp dirs")

    dataset = "qm9_std_jctc"

    dataset_file = os.path.join(save_path, dataset, dataset + "_data_energy.npy")
    if not os.path.exists(os.path.join(save_path, dataset)):
        os.makedirs(os.path.join(save_path, dataset))

    sorted_struct = sorted(all_struct, key=lambda x: len(x["Atoms"]))
    np.save(dataset_file, sorted_struct)

    shutil.rmtree(tmpdir, ignore_errors=True)

    print("Finished clearing temp dirs")
