"""
Torch datasets to handle IPFS data.
"""

from abc import ABC, abstractmethod
import pathlib
from typing import Any, Callable, Iterable, Optional, Union

import ipfshttpclient
import torch
from torch.utils.data import Dataset

from .parsers import IPFSImageTensorParser
from .utils import _download_item

class IPFSDatasetBase(Dataset, ABC):
    """
    Abstract base class for IPFS datasets.
    """

    def __init__(self,
                 ipfs_client: ipfshttpclient.Client,
                 data_folder: Union[pathlib.Path, str],
                 hashes: Iterable[str],
                 transform: Optional[Callable[torch.Tensor, torch.Tensor]] = None,
                 eager_download: bool = True,
                 suppress_errors: bool = False):
        """
        Initializes IPFSDatasetBase.

        Args:
            ipfs_client (ipfshttpclient.Client): An active IPFS client.
            data_folder (Union[pathlib.Path, str]): The path to the folder where the files
                will be downloaded.
            hashes (Iterable[str]): An iterable containing the hashes of the elements.
            transform (Optional[Callable[torch.Tensor, torch.Tensor]]): Transform that will be applied
                to tensors. If None, no transform is applied.
            eager_download (bool, optional): If True, elements are downloaded as soon as their hash is
                obtained. Defaults to True.
            suppress_errors (bool, optional): If True, errors during download will be ignored. Defaults
                to False.
        """
        self._hashes = hashes
        self._data = None

        self._ipfs_client = ipfs_client
        self._data_folder = pathlib.Path(data_folder)
        self._transform = transform

        self._eager_download = eager_download
        self._suppress_errors = suppress_errors

    @abstractmethod
    def init_data(self):
        """
        Initializes the dataset.
        """

    @abstractmethod
    def _download_item(self,
                       index: int):
        """
        Downloads a specific element with a given
        index.

        Args:
            index (int): The index of the element to download
        """

    def download_data(self,
                      indices: Optional[Union[int, Iterable[int], slice]] = None):
        """
        Downloads one or more elements with given indices.

        Args:
            indices (Optional[Union[int, Iterable[int], slice]], optional): The indices
                of the elements to download. If None, the entire dataset is downloaded.
                Defaults to None.
        """

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
                    indices: Union[int, Iterable[int], slice]):
        """
        Returns one or more elements of the dataset.

        Args:
            indices (Union[int, Iterable[int], slice]): The indices
                of the requested elements.

        Returns:
            Any: The requested elements.
        """
        self.download_data(indices=indices)

        item = self._data[indices]

        if self._transform is not None:
            item = self._transform(item)

        return item

    def __len__(self):
        if self._data is None:
            self.init_data()

        return len(self._data)


class IPFSGeneralDataset(IPFSDatasetBase):
    """
    A general-purpose IPFS dataset.
    """

    def __init__(self,
                 ipfs_client: ipfshttpclient.Client,
                 data_folder: Union[pathlib.Path, str],
                 hashes: Iterable[str],
                 transform: Optional[Callable[torch.Tensor, torch.Tensor]] = None,
                 parser: Optional[Callable[Any, Any]] = None,
                 eager_download: bool = True,
                 suppress_errors: bool = False):
        """
        Initializes IPFSGeneralDataset.

        Args:
            ipfs_client (ipfshttpclient.Client): An active IPFS client.
            data_folder (Union[pathlib.Path, str]): The path to the folder where the files
                will be downloaded.
            hashes (Iterable[str]): An iterable containing the hashes of the elements.
            transform (Optional[Callable[torch.Tensor, torch.Tensor]]): Transform that will be applied
                to tensors. If None, no transform is applied.
            parser (Optional[Callable[Any, Any]], optional): If not None, elements will be processed with this parser
                before being stored in memory. Defaults to None.
            eager_download (bool, optional): If True, elements are downloaded as soon as their hash is
                obtained. Defaults to True.
            suppress_errors (bool, optional): If True, errors during download will be ignored. Defaults
                to False.
        """
        super().__init__(ipfs_client, data_folder, hashes, transform=transform,
                         eager_download=eager_download, suppress_errors=suppress_errors)
        self._parser = parser

    def init_data(self):
        """
        Initializes the dataset.
        """
        self._data = [None] * len(self._hashes)
        if self._eager_download:
            self.download_data()

    def _download_item(self, index: int):
        """
        Downloads a specific element with a given
        index.

        Args:
            index (int): The index of the element to download
        """
        element_hash = self._hashes[index]
        file_contents = _download_item(
            element_hash, self._ipfs_client, self._data_folder)
        self._data[index] = file_contents if self._parser is None else self._parser(
            file_contents)


class IPFSTensorDataset(IPFSGeneralDataset):
    """
    A general-purpose tensor dataset.
    """

    def __init__(self,
                 ipfs_client: ipfshttpclient.Client,
                 data_folder: Union[pathlib.Path, str],
                 element_shape: Optional[Iterable[int]],
                 hashes: Iterable[str],
                 transform: Optional[Callable[torch.Tensor, torch.Tensor]] = None,
                 parser: Optional[Callable[Any, Any]] = None,
                 dtype: torch.dtype = torch.float32,
                 device: Union[torch.device, str] = 'cpu',
                 eager_download: bool = True,
                 suppress_errors: bool = False):
        """
        Initializes IPFSTensorDataset.

        Args:
            ipfs_client (ipfshttpclient.Client): An active IPFS client.
            data_folder (Union[pathlib.Path, str]): The path to the folder where the files
                will be downloaded.
            element_shape (Optional[Iterable[int]]): The shape of a tensor element. If None, no
                assumptions are made about the tensor shape.
            hashes (Iterable[str]): An iterable containing the hashes of the elements.
            transform (Optional[Callable[torch.Tensor, torch.Tensor]]): Transform that will be applied
                to tensors. If None, no transform is applied.
            parser (Optional[Callable[Any, Any]], optional): If not None, elements will be processed with this parser
                before being stored in memory. Defaults to None.
            dtype (torch.dtype, optional): The dtye of the tensors. Defaults to torch.float32.
            device (Union[torch.device, str], optional): The device on which tensors are stored. Defaults to 'cpu'.
            eager_download (bool, optional): If True, elements are downloaded as soon as their hash is
                obtained. Defaults to True.
            suppress_errors (bool, optional): If True, errors during download will be ignored. Defaults
                to False.
        """
        super().__init__(ipfs_client, data_folder, hashes, transform=transform,
                         parser=parser, eager_download=eager_download, suppress_errors=suppress_errors)

        self._element_shape = element_shape
        self._dtype = dtype
        self._device = device

    def init_data(self):
        """
        Initializes the dataset.
        """
        if self._element_shape is None:
            self._data = [None] * len(self._hashes)
        else:
            self._data = torch.zeros(
                (len(self._hashes),) + self._element_shape, dtype=self._dtype, device=self._device)


class IPFSImageTensorDataset(IPFSTensorDataset):
    """
    An image tensor dataset.
    """

    def __init__(self,
                 ipfs_client: ipfshttpclient.Client,
                 data_folder: Union[pathlib.Path, str],
                 image_shape: Optional[Iterable[int]],
                 hashes: Iterable[str],
                 transform: Optional[Callable[torch.Tensor, torch.Tensor]] = None,
                 channel_first: bool = True,
                 dtype: torch.dtype = torch.float32,
                 device: Union[torch.device, str] = 'cpu',
                 eager_download: bool = True,
                 suppress_errors: bool = False):
        """
        Initializes IPFSImageTensorDataset.

        Args:
            ipfs_client (ipfshttpclient.Client): An active IPFS client.
            data_folder (Union[pathlib.Path, str]): The path to the folder where the files
                will be downloaded.
            image_shape (Optional[Iterable[int]]): The shape of an image tensor. If None, no
                assumptions are made about the image shape.
            hashes (Iterable[str]): An iterable containing the hashes of the images.
            transform (Optional[Callable[torch.Tensor, torch.Tensor]]): Transform that will be applied
                to tensors. If None, no transform is applied.
            channel_first (bool) : If True, images will be stored in CHW format, else HWC. Defaults to True.
            dtype (torch.dtype, optional): The dtye of the image tensors. Defaults to torch.float32.
            device (Union[torch.device, str], optional): The device on which image tensors are stored. Defaults to 'cpu'.
            eager_download (bool, optional): If True, images are downloaded as soon as their hash is
                obtained. Defaults to True.
            suppress_errors (bool, optional): If True, errors during download will be ignored. Defaults
                to False.
        """
        super().__init__(ipfs_client, data_folder, image_shape, hashes, transform=transform, parser=IPFSImageTensorParser(channel_first=channel_first,
                                                                                                                          dtype=dtype, device=device), dtype=dtype, device=device, eager_download=eager_download, suppress_errors=suppress_errors)
