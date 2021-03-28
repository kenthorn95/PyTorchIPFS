import io
import logging
from typing import Any

import dataset
import ipfshttpclient

import torch

logger = logging.getLogger(__name__)

def _serialize(data : Any):
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

def _deserialize(serialized : bytes):
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

def store_checkpoint(ipfs_client : ipfshttpclient.Client, checkpoint : dict):
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

def get_checkpoint(ipfs_client : ipfshttpclient.Client, checkpoint_hash : str):
    """
    Retrieves a checkpoint from IPFS.

    Args:
        ipfs_client (ipfshttpclient.Client): The client that will be used to
            retrieve the checkpoint.
        checkpoint_hash (str): Hash of the checkpoint.

    Returns:
        dict: The retrieved checkpoint.
    """
    serialized = ipfs_client.get(checkpoint_hash)

    return _deserialize(serialized)

class CheckpointBackup:
    """
    Backups checkpoints on IPFS.
    """
    def __init__(self,
                ipfs_client : ipfshttpclient.Client,
                verbose : bool = True):
        """
        Initializes CheckpointBackup.

        Args:
            ipfs_client (ipfshttpclient.Client): [description]
            verbose (bool, optional): [description]. Defaults to True.
        """
        self._ipfs_client = ipfs_client
        self._verbose = verbose
        self.checkpoint_hashes = []

    def store_checkpoint(self, checkpoint):
        checkpoint_hash = store_checkpoint(self._ipfs_client, checkpoint)
        if self._verbose:
            logger.info('Storing checkpoint as %s.', checkpoint_hash)
        self.checkpoint_hashes.append(checkpoint_hash)

        return checkpoint_hash

    def get_checkpoint(self, checkpoint_hash):
        return get_checkpoint(self._ipfs_client, checkpoint_hash)

    @property
    def latest_checkpoint(self):
        if len(self.checkpoint_hashes) == 0:
            return None

        return self.get_checkpoint(self.checkpoint_hashes[-1])