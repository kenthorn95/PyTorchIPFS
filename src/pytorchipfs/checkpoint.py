"""
Tools and utilities to store and load checkpoints
on IPFS.
"""

import io
import logging
import pathlib
from typing import Any, Union

import ipfshttpclient
import torch

from .utils import _download_item

logger = logging.getLogger(__name__)


def _serialize(data: Any):
    """
    Serializes to bytes using PyTorch's
    serialization tools.

    Args:
        data (Any): The data to be serialized.

    Returns:
        bytes: The data serialized to bytes.
    """
    buff = io.BytesIO()
    torch.save(data, buff)
    buff.seek(0)
    return buff.read()


def _deserialize(serialized: bytes):
    """
    Deserializes from bytes using PyTorch's
    serialization tools.

    Args:
        serialized (bytes): The data as bytes.

    Returns:
        Any: The original data
    """
    buff = io.BytesIO()
    buff.write(serialized)
    buff.seek(0)
    return torch.load(buff)


def store_checkpoint(ipfs_client: ipfshttpclient.Client, checkpoint: dict):
    """
    Stores a checkpoint on IPFS.

    Args:
        ipfs_client (ipfshttpclient.Client): The client that will be used
            to store the checkpoint.
        checkpoint (dict): The checkpoint to be stored.

    Returns:
        str: The hash of the stored checkpoint.
    """
    serialized = _serialize(checkpoint)

    checkpoint_hash = ipfs_client.add_bytes(serialized)
    return checkpoint_hash


def get_checkpoint(ipfs_client: ipfshttpclient.Client, checkpoint_hash: str, data_folder: Union[str, pathlib.Path]):
    """
    Retrieves a checkpoint from IPFS.

    Args:
        ipfs_client (ipfshttpclient.Client): The client that will be used to
            retrieve the checkpoint.
        checkpoint_hash (str): Hash of the checkpoint.
        data_folder (Union[str, pathlib.Path]): The directory where the checkpoint
            will be downloaded.

    Returns:
        dict: The retrieved checkpoint.
    """
    serialized = _download_item(checkpoint_hash, ipfs_client, data_folder)

    return _deserialize(serialized)


class CheckpointBackup:
    """
    Backups checkpoints on IPFS.
    """

    def __init__(self,
                 ipfs_client: ipfshttpclient.Client,
                 data_folder: Union[str, pathlib.Path],
                 verbose: bool = True):
        """
        Initializes CheckpointBackup.

        Args:
            ipfs_client (ipfshttpclient.Client): The client that will be used to
                manage checkpoints.
            data_folder (Union[str, pathlib.Path]): The directory where the files
                will be downloaded.
            verbose (bool, optional): If True, logs checkpoint storage at INFO level.
                Defaults to True.
        """
        self._ipfs_client = ipfs_client
        self._data_folder = data_folder
        self._verbose = verbose
        self.checkpoint_hashes = []

    def store_checkpoint(self, checkpoint: dict):
        """
        Stores a checkpoint on IPFS.

        Args:
            checkpoint (dict): The checkpoint to be stored.

        Returns:
            str: The hash of the stored checkpoint.
        """
        checkpoint_hash = store_checkpoint(self._ipfs_client, checkpoint)
        if self._verbose:
            logger.info('Storing checkpoint as %s.', checkpoint_hash)
        self.checkpoint_hashes.append(checkpoint_hash)

        return checkpoint_hash

    def get_checkpoint(self, checkpoint_hash: str):
        """
        Retrieves a checkpoint from IPFS.

        Args:
            checkpoint_hash (str): Hash of the checkpoint.

        Returns:
            dict: The retrieved checkpoint.
        """
        return get_checkpoint(self._ipfs_client, checkpoint_hash, self._data_folder)

    @property
    def latest_checkpoint(self):
        if len(self.checkpoint_hashes) == 0:
            return None

        return self.get_checkpoint(self.checkpoint_hashes[-1])
