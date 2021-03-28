"""
Parsers to convert IPFS data to more usable formats.
"""

from abc import ABC, abstractmethod
import io
from typing import Any, Iterable, Union

from PIL import Image
import numpy as np
import torch

class IPFSParserBase(ABC):
    """
    Base class for IPFS parsers.
    """
    @abstractmethod
    def __call__(self, data : bytes):
        """
        Parses IPFS data.

        Args:
            data (bytes): The IPFS data.
        """

class SequentialParser(IPFSParserBase):
    """
    Chains multiple parsers in sequence.
    """
    def __init__(self,
                parsers : Iterable[IPFSParserBase]):
        """
        Initializes SequentialParser.

        Args:
            parsers (Iterable[IPFSParserBase]): The parsers, in
                order of execution.
        """
        self._parsers = list(parsers)

    def __call__(self, data : bytes):
        """
        Parses IPFS data.

        Args:
            data (bytes): The IPFS data.
        """
        for parser in self._parsers:
            data = parser(data)

        return data

class IPFSImageParser(IPFSParserBase):
    """
    Parses IPFS data as an image.
    """
    def __call__(self, data : bytes):
        """
        Parses IPFS image data.

        Args:
            data (bytes): The IPFS image data.
        """
        return Image.open(io.BytesIO(data))

class IPFSImageTensorParser(IPFSImageParser):
    """
    Parses IPFS as an image and converts to a tensor.
    """
    def __init__(self,
                channel_first : bool = True,
                dtype : torch.dtype = torch.float32,
                device : Union[torch.device, str] = 'cpu'):
        """
        Initializes IPFSImageTensorParser.

        Args:
            channel_first (bool, optional): If True, images will be stored in CHW format, else HWC. Defaults to True.
            dtype (torch.dtype, optional): The dtye of the image tensors. Defaults to torch.float32.
            device (Union[torch.device, str], optional): The device on which image tensors are stored. Defaults to 'cpu'.
        """
        self._channel_first = channel_first
        self._dtype = dtype
        self._device = device

    def __call__(self, data : bytes):
        """
        Parses IPFS image data and converts to a tensor.

        Args:
            data (bytes): The IPFS image data.
        """
        image = super().__call__(data)
        np_array = np.asarray(image)
        torch_tensor = torch.from_numpy(np_array).to(dtype=self._dtype, device=self._device)

        if self._channel_first and len(torch_tensor.shape) >= 3:
            torch_tensor = torch.transpose(torch_tensor, 0, -1)
        
        return torch_tensor