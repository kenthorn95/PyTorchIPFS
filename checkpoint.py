import io
import logging

import dataset
import ipfshttpclient

import torch

logger = logging.getLogger(__name__)

def multi_folder_dataset():
    pass

def _serialize(data):
    # Serialize to bytes
    buff = io.BytesIO()
    torch.save(data, buff)
    buff.seek(0)
    return buff.read()

def _deserialize(serialized):
    buff = io.BytesIO()
    buff.write(serialized)
    buff.seek(0)
    return torch.load(buff)

class CheckpointTracker:
    def __init__(self, ipfs_client : ipfshttpclient.Client, verbose : bool = True):
        self._ipfs_client = ipfs_client
        self._verbose = verbose
        self.checkpoint_hashes = []

    def store_checkpoint(self, checkpoint):
        serialized = _serialize(checkpoint)

        checkpoint_hash = self._ipfs_client.add_bytes(serialized)
        if self._verbose:
            logger.info('Storing checkpoint as %s.', checkpoint_hash)
        self.checkpoint_hashes.append(checkpoint_hash)

    def load_checkpoint(self, checkpoint_hash):
        serialized = self._ipfs_client.get(checkpoint_hash)

        return _deserialize(serialized)

    def latest_checkpoint(self):
        if len(self.checkpoint_hashes) == 0:
            return None

        return self.load_checkpoint(self.checkpoint_hashes[-1])

def from_hash(hash):
    pass

def load_weights(hash):
    pass

def store_checkpoint(model):
    pass