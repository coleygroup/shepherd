"""
ShEPhERD: Diffusing Shape, Electrostatics, and Pharmacophores for Drug Design.

A generative diffusion model (DDPM) framework.
"""

from importlib.metadata import PackageNotFoundError, version

try:  # noqa: SIM105
    __version__ = version("shepherd")
except PackageNotFoundError:
    pass

from .model_loader import load_model, get_model_info, clear_model_cache

__all__ = [
    "load_model", 
    "get_model_info",
    "clear_model_cache",
]
