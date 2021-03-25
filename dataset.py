

from abc import ABC
import pathlib

import ipfshttpclient
import torch
from torch.utils.data import Dataset
"""
Lazy loading vs eager loading vs background loading. Multiprocessing support?
IPFS directory explorer? --> ls
Store weights on IPFS? --> store/load state_dict
"""

class IPFSDatasetBase(Dataset, ABC):
    def __init__(self, ipfs_client, data_folder, hashes, download_policy='eager', error_policy='ignore'):
        if hashes is None:
            hashes = []
        self._hashes = hashes
        self._data = None

        self._ipfs_client = ipfs_client
        self._data_folder = pathlib.Path(data_folder)

        self._download_policy = download_policy
        self._error_policy = error_policy

        if not self._data_folder.exists():
            self._data_folder.mkdir()

    def init_data(self):
        raise NotImplementedError()

    def _download_item(self, index):
        raise NotImplementedError

    def download_data(self, indices=None):
        if indices is None:
            indices = range(len(self._hashes))

        if isinstance(indices, slice):
            indices = indices.indices(len(self._hashes))

        for index in indices:
            if self._data[index] is None:
                self._download_item(index)

    def __getitem__(self, indices):
       self.download_data(indices=indices)

       return self._data[indices]


class IPFSGeneralDataset(IPFSDatasetBase):
    def __init__(self, ipfs_client, data_folder, hashes, parser=None, download_policy='eager', error_policy='ignore'):
        super().__init__(ipfs_client, data_folder, hashes, download_policy=download_policy, error_policy=error_policy)
        self._parser = parser

    def _init_data(self):
        self._data = [None] * len(self._hashes)
        if self._download_policy == 'eager':
            self.download_data()

    def _download_item(self, index):
        element_hash = self._hashes[index]
        element_path = self._data_folder / str(element_hash)

        if not element_path.exists():
            self._ipfs_client.get(element_hash, target=self._data_folder)

        with open(element_path, 'rb') as f:
            file_contents = f.read()
        self._data[index] = file_contents if self._parser is None else self._parser(file_contents)


class IPFSTensorDataset(IPFSGeneralDataset):
    def __init__(self, ipfs_client, data_folder, hashes, element_shape, parser=None, dtype=torch.float32, device='cpu', download_policy='eager', error_policy='ignore'):
        super().__init__(ipfs_client, data_folder, hashes, parser=parser, download_policy=download_policy, error_policy=error_policy)
        self._element_shape = element_shape
        self._dtype = dtype
        self._device = device

    def _init_data(self):
        self._data = torch.zeros((len(self._hashes),) + self._element_shape, dtype=self._dtype, device=self._device)