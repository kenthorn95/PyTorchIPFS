"""
Utility functions for IPFS.
"""

import pathlib
from typing import Union

import ipfshttpclient


def _download_item(element_hash: str,
                   ipfs_client: ipfshttpclient.Client,
                   data_folder: Union[str, pathlib.Path]):
    """
    Downloads a file from IPFS.

    Args:
        element_hash (str): Hash of the requested item.
        ipfs_client (ipfshttpclient.Client): An active IPFS client.
        data_folder (Union[str, pathlib.Path]): The path to the folder where the files
                will be downloaded.

    Returns:
        bytes: Content of the downloaded file.
    """
    data_folder = pathlib.Path(data_folder)

    if not data_folder.exists():
        data_folder.mkdir()

    element_path = data_folder / str(element_hash)

    if not element_path.exists():
        ipfs_client.get(element_hash, target=data_folder)

    with open(element_path, 'rb') as f:
        return f.read()
