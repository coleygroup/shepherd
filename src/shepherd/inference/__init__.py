"""
Shepherd Inference submodule.

Provides the main inference sampling function.
"""

from shepherd.inference.inference_original import inference_sample
from shepherd.inference.sampler import generate, generate_from_intermediate_time

__all__ = ['inference_sample', 'generate', 'generate_from_intermediate_time']