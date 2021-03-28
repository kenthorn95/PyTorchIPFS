from abc import ABC, abstractmethod
import io
from PIL import Image
from typing import Union

import numpy as np
import torch

class IPFSParserBase(ABC):
    @abstractmethod
    def __call__(self, data : bytes):
        raise NotImplementedError

class IPFSImageParser(IPFSParserBase):
    def __call__(self, data : bytes):
        return Image.open(io.BytesIO(data))

class IPFSImageTensorParser(IPFSImageParser):
    def __init__(self,
    channel_first : bool = True,
    dtype : torch.dtype = torch.float32,
    device : Union[torch.device, str] = 'cpu'):
        self._channel_first = channel_first
        self._dtype = dtype
        self._device = device

    def __call__(self, data : bytes):
        image = super().__call__(data)
        np_array = np.asarray(image)
        torch_tensor = torch.from_numpy(np_array).to(dtype=self._dtype, device=self._device)

        if self._channel_first and len(torch_tensor.shape) >= 3:
            torch_tensor = torch.transpose(torch_tensor, 0, -1)
        
        return torch_tensor