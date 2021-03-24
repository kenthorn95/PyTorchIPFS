

from abc import ABC

import ipfshttpclient
import torch
from torch.utils.data import Dataset
"""
Lazy loading vs eager loading vs background loading. Multiprocessing support?
IPFS directory explorer? --> ls
Store weights on IPFS? --> store/load state_dict
"""

class IPFSDatasetBase(Dataset, ABC):
    def __init__(self, ipfs_client, data_folder, hashes, downloading_policy='eager'):
        if hashes is None:
            hashes = []
        self.hashes = hashes
        self.data = None

        self.ipfs_client = ipfs_client
        self.data_folder = data_folder
        self.downloading_policy = downloading_policy

        if downloading_policy == 'eager':
            self.download_data()

    def download_data(self, indices=None):
        if indices is None:
            indices = range(len(self.hashes))

        if isinstance(indices, slice):
            indices = indices.indices(len(self.hashes))

        for index in indices:
            if self.data[index] is None:
                self._download_item(index)

    def __getitem__(self, indices) -> T_co:
       self.download_data(indices=indices)

       return self.data[indices]


class IPFSGeneralDataset(IPFSDatasetBase):
    def _download_item(self, index):
        ipfshttpclient.connect().get()

    

class IPFSTensorDataset(IPFSDatasetBase):
    def __init__(self, ipfs_client, data_folder, hashes, parser, downloading_policy='eager'):