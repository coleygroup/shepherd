"""
ShEPhERD: Diffusing Shape, Electrostatics, and Pharmacophores for Drug Design.

A generative diffusion model (DDPM) framework.
"""

from importlib.metadata import PackageNotFoundError, version

try:  # noqa: SIM105
    __version__ = version("shepherd")
except PackageNotFoundError:
    pass