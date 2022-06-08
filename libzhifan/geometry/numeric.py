from typing import Union, Callable
import numpy as np
import torch


def nptify(x) -> Callable:
    """
    Make a type convertor based on the type of x

    Example:
    a = torch.Tensor([1])
    b = np.array([])
    To convert b to the type of a:
    b_out = nptify(a)(b)

    Returns:
        A Callable that converts its input to x's type
    """
    if isinstance(x, torch.Tensor):
        return lambda a: torch.as_tensor(a, dtype=x.dtype, device=x.device)
    elif isinstance(x, np.ndarray):
        return lambda a: np.asarray(a)


def numpify(tensor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().squeeze().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor

