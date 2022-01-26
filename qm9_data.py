import logging
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib import request as request
from urllib.error import HTTPError, URLError
from base64 import b64encode, b64decode

import numpy as np
import torch
from ase.db import connect
from ase.io.extxyz import read_xyz
from ase.units import Debye, Bohr, Hartree, eV

from schnetpack import Properties
from schnetpack.datasets import DownloadableAtomsData
from utility_classes import ConnectivityCompressor
from preprocess_dataset import preprocess_dataset


class QM9gen(DownloadableAtomsData):
    """ QM9 benchmark dataset for organic molecules with up to nine non-hydrogen atoms
        from {C, O, N, F}.

        This class adds convenience functions to download QM9 from figshare,
        pre-process the data such that it can be used for moleculec generation with the
        G-SchNet model, and load the data into pytorch.

        Args:
            path (str): path to directory containing qm9 database
            subset (list, optional): indices of subset, set to None for entire dataset
                (default: None).
            download (bool, optional): enable downloading if qm9 database does not
                exists (default: True)
            precompute_distances (bool, optional): if True and the pre-processed
                database does not yet exist, the pairwise distances of atoms in the
                dataset's molecules will be computed during pre-processing and stored in
                the database (increases storage demand of the dataset but decreases
                computational cost during training as otherwise the distances will be
                computed once in every epoch, default: True)
            remove_invalid (bool, optional): if True QM9 molecules that do not pass the
                valence check will be removed from the training data (note 1: the
                validity is per default inferred from a pre-computed list in our
                repository but will be assessed locally if the download fails,
                note2: only works if the pre-processed database does not yet exist,
                default: True)

        References:
            .. [#qm9_1] https://ndownloader.figshare.com/files/3195404
    """

    # general settings for the dataset
    available_atom_types = [1, 6, 7, 8, 9]  # all atom types found in the dataset
    atom_types_valence = [1, 4, 3, 2, 1]  # valence constraints of the atom types
    radial_limits = [0.9, 1.7]  # minimum and maximum distance between neighboring atoms

    # properties
    A = 'rotational_constant_A'
    B = 'rotational_constant_B'
    C = 'rotational_constant_C'
    mu = 'dipole_moment'
    alpha = 'isotropic_polarizability'
    homo = 'homo'
    lumo = 'lumo'
    gap = 'gap'
    r2 = 'electronic_spatial_extent'
    zpve = 'zpve'
    U0 = 'energy_U0'
    U = 'energy_U'
    H = 'enthalpy_H'
    G = 'free_energy'
    Cv = 'heat_capacity'

    properties = [
        A, B, C, mu, alpha,
        homo, lumo, gap, r2, zpve,
        U0, U, H, G, Cv, 'n_atoms', 'relative_atomic_energy'
    ]

    units = [1., 1., 1., Debye, Bohr ** 3,
             Hartree, Hartree, Hartree,
             Bohr ** 2, Hartree,
             Hartree, Hartree, Hartree,
             Hartree, 1.,
            ]

    units_dict = dict(zip(properties, units))

    connectivity_compressor = ConnectivityCompressor()

    def __init__(self, path, subset=None, download=True, precompute_distances=True,
                 remove_invalid=True, load_additionally=None):
        self.path = path
        self.dbpath = os.path.join(self.path, f'qm9gen.db')
        self.precompute_distances = precompute_distances
        self.remove_invalid = remove_invalid
        self.load_additionally = [] if load_additionally is None else load_additionally
        self.db_metadata = None
        # hard coded weights for regression of energy per atom from the concentration of atoms (i.e. the composition divided by the absolute number of atoms)
        self.energy_regression = lambda x: x.dot(np.array([-1032.61411992, -2052.26704777, -2506.01057452, -3063.2081989, -3733.79421864])) + 1016.2017264092275

        super().__init__(self.dbpath, subset=subset,
                         available_properties=self.properties,
                         units=self.units, download=download)

        with connect(self.dbpath) as db:
            if db.count() <= 0:
                logging.error('Error: Data base is empty, please provide a path '
                              'to a proper data base or an empty path where the '
                              'data can be downloaded to!')
                raise FileExistsError()
            if self.precompute_distances and 'dists' not in db.get(1).data:
                logging.info('Caution: Existing data base does not contain '
                             'pre-computed distances, distances will be computed '
                             'on the fly in every epoch.')
            if 'fingerprint' in self.load_additionally:
                if 'fingerprint' not in db.get(1).data:
                    logging.error('Error: Fingerprints not found in the provided data '
                                  'base, please provide another path to the correct '
                                  'data base or an empty directory where a new data '
                                  'base with pre-computed fingerprints can be '
                                  'downloaded to.')
                    raise FileExistsError()
                else:
                    self.db_metadata = db.metadata
            for prop in self.load_additionally:
                if prop not in db.get(1).data and prop not in self.properties:
                    logging.error(f'Error: Unknown property ({prop}) requested for '
                                  f'conditioning. Cannot obtain that property from '
                                  f'the database!')
                    raise FileExistsError()

    def create_subset(self, idx):
        """
        Returns a new dataset that only consists of provided indices.

        Args:
            idx (numpy.ndarray): subset indices

        Returns:
            schnetpack.data.AtomsData: dataset with subset of original data
        """
        idx = np.array(idx)
        subidx = idx if self.subset is None or len(idx) == 0 \
            else np.array(self.subset)[idx]
        return type(self)(self.path, subidx, download=False,
                          load_additionally=self.load_additionally)

    def get_properties(self, idx):
        _idx = self._subset_index(idx)
        with connect(self.dbpath) as db:
            row = db.get(_idx + 1)
        at = row.toatoms()

        # extract/calculate structure
        properties = {}
        properties[Properties.Z] = torch.LongTensor(at.numbers.astype(np.int))
        positions = at.positions.astype(np.float32)
        positions -= at.get_center_of_mass()  # center positions
        properties[Properties.R] = torch.FloatTensor(positions)
        properties[Properties.cell] = torch.FloatTensor(at.cell.astype(np.float32))

        # recover connectivity matrix from compressed format
        con_mat = self.connectivity_compressor.decompress(row.data['con_mat'])
        # save in dictionary
        properties['_con_mat'] = torch.FloatTensor(con_mat.astype(np.float32))

        # extract pre-computed distances (if they exist)
        if 'dists' in row.data:
            properties['dists'] = row.data['dists']

        # extract additional information
        for add_prop in self.load_additionally:
            if add_prop == 'fingerprint':
                fp = np.array(row.data[add_prop],
                              dtype=self.db_metadata['fingerprint_format'])
                properties[add_prop] = torch.FloatTensor(
                    np.unpackbits(fp.view(np.uint8), bitorder='little'))
            elif add_prop == 'n_atoms':
                properties['n_atoms'] = torch.FloatTensor([len(at.numbers)])
            elif add_prop == 'relative_atomic_energy':
                types = at.numbers.astype(np.int)
                composition = np.array([np.sum(types == 1),
                                        np.sum(types == 6),
                                        np.sum(types == 7),
                                        np.sum(types == 8),
                                        np.sum(types == 9)],
                                       dtype=np.float32)
                concentration = composition/np.sum(composition)
                energy = row.data['energy_U0']
                energy_per_atom = energy/len(types)
                relative_atomic_energy = energy_per_atom - self.energy_regression(concentration)
                properties[add_prop] = torch.FloatTensor([relative_atomic_energy])
            else:
                properties[add_prop] = torch.FloatTensor([row.data[add_prop]])

        # get atom environment
        nbh_idx, offsets = self.environment_provider.get_environment(at)
        # store neighbors, cell, and index
        properties[Properties.neighbors] = torch.LongTensor(nbh_idx.astype(np.int))
        properties[Properties.cell_offset] = torch.FloatTensor(
            offsets.astype(np.float32))
        properties["_idx"] = torch.LongTensor(np.array([idx], dtype=np.int))

        return at, properties

    def _download(self):
        works = True
        if not os.path.exists(self.dbpath):
            qm9_path = os.path.join(self.path, f'qm9.db')
            if not os.path.exists(qm9_path):
                works = works and self._load_data()
            works = works and self._preprocess_qm9()
        return works

    def _load_data(self):
        logging.info('Downloading GDB-9 data...')
        tmpdir = tempfile.mkdtemp('gdb9')
        tar_path = os.path.join(tmpdir, 'gdb9.tar.gz')
        raw_path = os.path.join(tmpdir, 'gdb9_xyz')
        url = 'https://ndownloader.figshare.com/files/3195389'

        try:
            request.urlretrieve(url, tar_path)
            logging.info('Done.')
        except HTTPError as e:
            logging.error('HTTP Error:', e.code, url)
            return False
        except URLError as e:
            logging.error('URL Error:', e.reason, url)
            return False

        logging.info('Extracting data from tar file...')
        tar = tarfile.open(tar_path)
        tar.extractall(raw_path)
        tar.close()
        logging.info('Done.')

        logging.info('Parsing xyz files...')
        with connect(os.path.join(self.path, 'qm9.db')) as db:
            ordered_files = sorted(os.listdir(raw_path),
                                   key=lambda x: (int(re.sub('\D', '', x)), x))
            for i, xyzfile in enumerate(ordered_files):
                xyzfile = os.path.join(raw_path, xyzfile)

                if (i + 1) % 10000 == 0:
                    logging.info('Parsed: {:6d} / 133885'.format(i + 1))
                properties = {}
                tmp = os.path.join(tmpdir, 'tmp.xyz')

                with open(xyzfile, 'r') as f:
                    lines = f.readlines()
                    l = lines[1].split()[2:]
                    for pn, p in zip(self.properties, l):
                        properties[pn] = float(p) * self.units[pn]
                    with open(tmp, "wt") as fout:
                        for line in lines:
                            fout.write(line.replace('*^', 'e'))

                with open(tmp, 'r') as f:
                    ats = list(read_xyz(f, 0))[0]
                db.write(ats, data=properties)
        logging.info('Done.')

        shutil.rmtree(tmpdir)

        return True

    def _preprocess_qm9(self):
        # try to download pre-computed list of invalid molecules
        raw_path = os.path.join(self.path, 'qm9_invalid.txt')
        if os.path.exists(raw_path):
            logging.info(f'Found existing list with indices of molecules in QM9 that are invalid at "{raw_path}".'
                f' Please manually delete the file and restart training if you want to use the default list instead.')
            invalid_list = np.loadtxt(raw_path)
        else:
            logging.info('Downloading pre-computed list of invalid QM9 molecules...')
            # url = 'https://github.com/atomistic-machine-learning/cG-SchNet/blob/main/splits/' \
            #       'qm9_invalid.txt?raw=true'
            try:
                url = Path(__file__).parent.resolve() / 'splits/qm9_invalid.txt'
                request.urlretrieve(url.as_uri(), raw_path)
                logging.info('Done.')
                invalid_list = np.loadtxt(raw_path)
            except HTTPError as e:
                logging.error('HTTP Error:', e.code, url)
                logging.info('CAUTION: Could not download pre-computed list, will assess '
                             'validity during pre-processing.')
                invalid_list = None
            except URLError as e:
                logging.error('URL Error:', e.reason, url)
                logging.info('CAUTION: Could not download pre-computed list, will assess '
                             'validity during pre-processing.')
                invalid_list = None
            except ValueError as e:
                logging.error('Value Error:', e)
                logging.info('CAUTION: Could not download pre-computed list, will assess '
                             'validity during pre-processing.')
                invalid_list = None
        # check validity of molecules and store connectivity matrices and interatomic
        # distances in database as a pre-processing step
        qm9_db = os.path.join(self.path, f'qm9.db')
        valence_list = \
            np.array([self.available_atom_types, self.atom_types_valence]).flatten('F')
        precompute_fingerprint = 'fingerprint' in self.load_additionally
        preprocess_dataset(datapath=qm9_db, valence_list=list(valence_list),
                           n_threads=8, n_mols_per_thread=125, logging_print=True,
                           new_db_path=self.dbpath,
                           precompute_distances=self.precompute_distances,
                           precompute_fingerprint=precompute_fingerprint,
                           remove_invalid=self.remove_invalid,
                           invalid_list=invalid_list)
        return True

    def get_available_properties(self, available_properties):
        # we don't use properties other than stored connectivity matrices (and
        # distances, if they were precomputed) so we skip this part
        return available_properties
        
    def create_splits(self, num_train=None, num_val=None, split_file=None):
        """
        Splits the dataset into train/validation/test splits, writes split to
        an npz file and returns subsets. Either the sizes of training and
        validation split or an existing split file with split indices have to
        be supplied. The remaining data will be used in the test dataset.
        Args:
            num_train (int): number of training examples
            num_val (int): number of validation examples
            split_file (str): Path to split file. If file exists, splits will
                              be loaded. Otherwise, a new file will be created
                              where the generated split is stored.
        Returns:
            schnetpack.data.AtomsData: training dataset
            schnetpack.data.AtomsData: validation dataset
            schnetpack.data.AtomsData: test dataset
        """
        invalid_file_path = os.path.join(self.path, f"qm9_invalid.txt")
        if not os.path.exists(invalid_file_path):
            raise ValueError(f"Cannot find required file with indices of QM9 molecules that are invalid at {invalid_file_path}!")
        removed_idx = {*np.loadtxt(invalid_file_path).astype(int)}
        if split_file is not None and os.path.exists(split_file):
            S = np.load(split_file)
            train_idx = S["train_idx"].tolist()
            val_idx = S["val_idx"].tolist()
            test_idx = S["test_idx"].tolist()
            invalid_idx = {*S["invalid_idx"]}
        else:
            if num_train is None or num_val is None:
                raise ValueError(
                    "You have to supply either split sizes (num_train /"
                    + " num_val) or an npz file with splits."
                )

            assert num_train + num_val <= len(
                self
            ), "Dataset is smaller than num_train + num_val!"

            num_train = num_train if num_train > 1 else num_train * len(self)
            num_val = num_val if num_val > 1 else num_val * len(self)
            num_train = int(num_train)
            num_val = int(num_val)

            idx = np.random.permutation(len(self))
            train_idx = idx[:num_train].tolist()
            val_idx = idx[num_train : num_train + num_val].tolist()
            test_idx = idx[num_train + num_val :].tolist()
            invalid_idx = removed_idx

            if split_file is not None:
                np.savez(
                    split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, 
                    invalid_idx=sorted(list(invalid_idx))
                )
        
        if len(removed_idx) != len(invalid_idx) or len(removed_idx.difference(invalid_idx)) != 0:
            raise ValueError(f"Mismatch between the data base used to generate the provided split file and your local database. "
                + f"Please specify an empty data directory to re-download QM9 and try again.")
        train = self.create_subset(train_idx)
        val = self.create_subset(val_idx)
        test = self.create_subset(test_idx)
        return train, val, test
