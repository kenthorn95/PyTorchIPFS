

from abc import ABC, abstractmethod
import pathlib
from typing import Any, Callable, Iterable, Optional, Union

import ipfshttpclient
import torch
from torch.utils.data import Dataset
"""
Lazy loading vs eager loading vs background loading. Multiprocessing support?
IPFS directory explorer? --> ls
"explorer" provides utility functions
"""

class IPFSDatasetBase(Dataset, ABC):
    def __init__(self,
        ipfs_client : ipfshttpclient.Client,
        data_folder : Union[pathlib.Path, str],
        hashes : Iterable[str],
        eager_download : bool = True,
        suppress_errors : bool = False):
        """
        Abstract base class for IPFS datasets.

        Args:
            ipfs_client (ipfshttpclient.Client): [description]
            data_folder (Union[pathlib.Path, str]): [description]
            hashes (Iterable[str]): [description]
            eager_download (bool, optional): [description]. Defaults to True.
            suppress_errors (bool, optional): [description]. Defaults to False.
        """
        if hashes is None:
            hashes = []
        self._hashes = hashes
        self._data = None

        self._ipfs_client = ipfs_client
        self._data_folder = pathlib.Path(data_folder)

        self._eager_download = eager_download
        self._suppress_errors = suppress_errors

        if not self._data_folder.exists():
            self._data_folder.mkdir()

    @abstractmethod
    def init_data(self):
        pass

    @abstractmethod
    def _download_item(self,
                       index : int):
        """
        Downloads a specific item with a given
        index.

        Args:
            index (int): The index of the element to download
        """

    def download_data(self,
                      indices : Iterable[int] = None):
        if self._data is None:
            self.init_data()

        if indices is None:
            indices = list(range(len(self._hashes)))

        if isinstance(indices, slice):
            indices = indices.indices(len(self._hashes))
        if isinstance(indices, int):
            indices = [indices]

        for index in indices:
            if self._data[index] is None:
                try:
                    self._download_item(index)
                except:
                    if not self._suppress_errors:
                        raise

    def __getitem__(self,
                    indices : Union[int, Iterable[int], slice]):
        """
        Returns one or more elements of the dataset.

        Args:
            indices (Union[int, Iterable[int], slice]): The indices
                of the requested elements.

        Returns:
            Any: The requested elements.
        """
        self.download_data(indices=indices)

        return self._data[indices]


class IPFSGeneralDataset(IPFSDatasetBase):
    def __init__(self,
        ipfs_client : ipfshttpclient.Client,
        data_folder : Union[pathlib.Path, str],
        hashes : Iterable[str],
        parser : Optional[Callable[Any, Any]] = None,
        eager_download : bool = True,
        suppress_errors : bool = False):
        super().__init__(ipfs_client, data_folder, hashes, eager_download=eager_download, suppress_errors=suppress_errors)
        self._parser = parser

    def init_data(self):
        self._data = [None] * len(self._hashes)
        if self._eager_download:
            self.download_data()

    def _download_item(self, index : int):
        element_hash = self._hashes[index]
        element_path = self._data_folder / str(element_hash)

        if not element_path.exists():
            self._ipfs_client.get(element_hash, target=self._data_folder)

        with open(element_path, 'rb') as f:
            file_contents = f.read()
        self._data[index] = file_contents if self._parser is None else self._parser(file_contents)


class IPFSTensorDataset(IPFSGeneralDataset):
    def __init__(self,
        ipfs_client : ipfshttpclient.Client,
        data_folder : Union[pathlib.Path, str],
        element_shape : Iterable[int],
        hashes : Iterable[str],
        parser : Optional[Callable[Any, Any]] = None,
        dtype : torch.dtype = torch.float32,
        device : Union[torch.device, str] = 'cpu',
        eager_download : bool = True,
        suppress_errors : bool = False):
        super().__init__(ipfs_client, data_folder, hashes, parser=parser, eager_download=eager_download, suppress_errors=suppress_errors)

        self._element_shape = element_shape
        self._dtype = dtype
        self._device = device

    def _init_data(self):
        self._data = torch.zeros((len(self._hashes),) + self._element_shape, dtype=self._dtype, device=self._device)