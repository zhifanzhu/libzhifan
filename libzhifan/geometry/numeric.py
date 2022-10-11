from functools import singledispatch
from typing import Union, Callable
import numpy as np
import torch


@singledispatch
def nptify(x) -> Callable:
    """ Num-Py-Torch-i-FYing
    Make a type convertor based on the type of x

    Example:
    a = torch.Tensor([1])
    b = np.array([])
    To convert b to the type of a:
    b_out = nptify(a)(b)

    Returns:
        A Callable that converts its input to x's type
    """
    raise NotImplementedError

@nptify.register
def _nptify(x: torch.Tensor):
    return lambda a: torch.as_tensor(a, dtype=x.dtype, device=x.device)
@nptify.register
def _nptify(x: np.ndarray):
    return lambda a: np.asarray(a)


@singledispatch
def numpize(tensor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    raise NotImplementedError

@numpize.register
def _numpize(tensor: torch.Tensor):
    return tensor.detach().squeeze().cpu().numpy()
@numpize.register
def _numpize(tensor: np.ndarray):
    return tensor

