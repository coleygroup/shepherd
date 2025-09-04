"""
Standalone model loader for ShEPhERD that works outside of Streamlit.
"""
import torch
from typing import Literal, Optional
from pathlib import Path

from shepherd.lightning_module import LightningModule
from shepherd.checkpoint_manager import get_checkpoint_path


def load_model(
    model_type: Literal['mosesaq', 'gdb_x2', 'gdb_x3', 'gdb_x4'] = 'mosesaq',
    device: Optional[str] = None,
    local_data_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
) -> LightningModule:
    """
    Load a ShEPhERD model with automatic checkpoint downloading.
    
    This function provides a clean interface for loading ShEPhERD models
    that works both in research environments and production deployments.
    
    Arguments
    ---------
    model_type: Type of model to load
        - 'mosesaq': MOSES-aq trained with shape, electrostatics, and pharmacophores
        - 'gdb_x2': GDB17 trained with shape conditioning
        - 'gdb_x3': GDB17 trained with shape and electrostatics  
        - 'gdb_x4': GDB17 trained with pharmacophores
    device: Device to load model on ('cuda', 'cpu', or None for auto-detection)
    local_data_dir: Directory containing local checkpoints (for backward compatibility)
    cache_dir: Directory to cache downloaded checkpoints (None uses default HF cache)
    force_download: Whether to force download even if local checkpoint exists
    verbose: Whether to print verbose output

    Returns
    -------
    Loaded and initialized ShEPhERD model ready for inference
        
    Example
    -------
    >>> # Load default MOSES-aq model
    >>> model = load_model()
    
    >>> # Load specific model type on GPU
    >>> model = load_model('gdb_x3', device='cuda')
    
    >>> # Force download latest version
    >>> model = load_model('mosesaq', force_download=True)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Get checkpoint path with automatic downloading
        model_path = get_checkpoint_path(
            model_type=model_type,
            local_data_dir=local_data_dir,
            cache_dir=cache_dir,
            force_download=force_download
        )

        print(f"Loading {model_type} model from: {model_path}")
        print(f"Using device: {device}")

        device_obj = torch.device(device)
        model_pl = LightningModule.load_from_checkpoint(
            model_path,
            weights_only=True,
            map_location=device_obj
        )

        model_pl.eval()
        model_pl.model.device = device_obj

        print(f"Successfully loaded {model_type} model")
        return model_pl

    except Exception as e:
        raise RuntimeError(f"Failed to load {model_type} model: {str(e)}") from e


def get_model_info() -> dict:
    """
    Get information about available ShEPhERD models.
    
    Returns
    -------
    Dictionary mapping model types to their descriptions
    """
    from shepherd.checkpoint_manager import CheckpointManager

    manager = CheckpointManager()
    return manager.get_available_models()


def clear_model_cache(model_type: Optional[str] = None, cache_dir: Optional[str] = None):
    """
    Clear cached model checkpoints.
    
    Arguments
    ---------
    model_type: Specific model type to clear, or None to clear all
    cache_dir: Cache directory to clear from (None uses default HF cache)
    """
    from shepherd.checkpoint_manager import CheckpointManager

    manager = CheckpointManager(cache_dir=cache_dir)
    manager.clear_cache(model_type=model_type)
