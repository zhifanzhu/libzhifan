import torch

def gather_ext(src: torch.Tensor, 
               index: torch.Tensor, 
               dim: int) -> torch.Tensor:
    """
    Args:
        src: shape (dim0, ..., dim_d, ..., dim_k, ..., dim_n)
        index: shape (dim0, ..., dim_d, ..., dim_k)
        dim: int
    Returns:
        gathered: (dim0, dim1, ..., dim_n)
    """    
    index = index.expand_as(src)
    return src.gather(dim=dim, index=index)