"""
Checkpoint manager for downloading and caching ShEPhERD model weights from Hugging Face.
"""
import os
import torch
from pathlib import Path
from typing import Literal, Optional
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.utils import HfHubHTTPError

HF_REPO_ID = 'kabeywar/shepherd'

# Model checkpoint mappings
MODEL_CHECKPOINTS = {
    'mosesaq': {
        'filename': 'x1x3x4_diffusion_mosesaq_20240824_submission.ckpt',
        'repo_id': HF_REPO_ID,
        'description': 'MOSES-aq trained model with shape, electrostatics, and pharmacophores'
    },
    'gdb_x2': {
        'filename': 'x1x2_diffusion_gdb17_20240824_submission.ckpt', 
        'repo_id': HF_REPO_ID,
        'description': 'GDB17 trained model with shape conditioning'
    },
    'gdb_x3': {
        'filename': 'x1x3_diffusion_gdb17_20240824_submission.ckpt',
        'repo_id': HF_REPO_ID,
        'description': 'GDB17 trained model with shape and electrostatics'
    },
    'gdb_x4': {
        'filename': 'x1x4_diffusion_gdb17_20240824_submission.ckpt',
        'repo_id': HF_REPO_ID,
        'description': 'GDB17 trained model with pharmacophores'
    }
}


class CheckpointManager:
    """Manages downloading and caching of ShEPhERD model checkpoints."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize checkpoint manager.
        
        Arguments
        ---------
        cache_dir: Directory to cache checkpoints. If None, uses default HF cache.
            Typically this is: ~/.cache/huggingface/
        """
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def get_checkpoint_path(
        self, 
        model_type: Literal['mosesaq', 'gdb_x2', 'gdb_x3', 'gdb_x4'],
        force_download: bool = False
    ) -> str:
        """
        Download or retrieve cached checkpoint for the specified model type.
        
        Arguments
        ---------
        model_type: Type of model checkpoint to retrieve
        force_download: Whether to force re-download even if cached
            
        Returns
        -------
        Path to the downloaded checkpoint file
        """
        if model_type not in MODEL_CHECKPOINTS:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available types: {list(MODEL_CHECKPOINTS.keys())}")

        checkpoint_info = MODEL_CHECKPOINTS[model_type]

        try:
            # Download from Hugging Face Hub with caching
            checkpoint_path = hf_hub_download(
                repo_id=checkpoint_info['repo_id'],
                filename=checkpoint_info['filename'],
                repo_type='model',
                cache_dir=self.cache_dir,
                force_download=force_download
            )

            # Verify the file exists and is readable
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Downloaded checkpoint not found at {checkpoint_path}")

            return checkpoint_path

        except HfHubHTTPError as e:
            if "401" in str(e):
                raise HfHubHTTPError(
                    f"Access denied to {checkpoint_info['repo_id']}. "
                    "The repository might be private or require authentication. "
                    "Please check the repository permissions or provide a valid token."
                ) from e
            elif "404" in str(e):
                raise HfHubHTTPError(
                    f"Checkpoint not found: {checkpoint_info['filename']} "
                    f"in repository {checkpoint_info['repo_id']}. "
                    "Please verify the repository and file names are correct."
                ) from e
            else:
                raise HfHubHTTPError(
                    f"Failed to download checkpoint: {e}. "
                    "Please check your internet connection and try again."
                ) from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error downloading checkpoint: {e}") from e

    def check_local_checkpoint(
        self, 
        model_type: Literal['mosesaq', 'gdb_x2', 'gdb_x3', 'gdb_x4'],
        local_path: str
    ) -> bool:
        """
        Check if a local checkpoint file exists and is valid.
        
        Arguments
        ---------
        model_type: Type of model checkpoint
        local_path: Path to check for local checkpoint
            
        Returns
        -------
        True if local checkpoint exists and appears valid
        """
        if not os.path.exists(local_path):
            return False

        try:
            # Try to load checkpoint metadata to verify it's a valid PyTorch checkpoint
            checkpoint = torch.load(local_path, map_location='cpu', weights_only=False)

            # Basic validation - check if it has expected keys
            required_keys = ['state_dict', 'hyper_parameters']
            has_required_keys = all(key in checkpoint for key in required_keys)

            return has_required_keys

        except Exception:
            return False

    def get_available_models(self) -> dict:
        """
        Get information about available model checkpoints.
        
        Returns
        -------
        Dictionary with model types and their descriptions
        """
        return {
            model_type: info['description'] 
            for model_type, info in MODEL_CHECKPOINTS.items()
        }

    def clear_cache(self, model_type: Optional[str] = None):
        """
        Clear cached checkpoints.
        
        Arguments
        ---------
        model_type: Specific model type to clear, or None to clear all
        """
        if not self.cache_dir:
            print("Using default HF cache directory. Use huggingface-cli to manage cache.")
            return

        if model_type:
            if model_type not in MODEL_CHECKPOINTS:
                raise ValueError(f"Unknown model type: {model_type}")
            # Clear specific model
            filename = MODEL_CHECKPOINTS[model_type]['filename']
            cache_path = os.path.join(self.cache_dir, filename)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                print(f"Cleared cache for {model_type}")
        else:
            # Clear all cached checkpoints
            for model_info in MODEL_CHECKPOINTS.values():
                cache_path = os.path.join(self.cache_dir, model_info['filename'])
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            print("Cleared all cached checkpoints")


def get_checkpoint_path(
    model_type: Literal['mosesaq', 'gdb_x2', 'gdb_x3', 'gdb_x4'],
    local_data_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False
) -> str:
    """
    Convenience function to get checkpoint path with fallback to local files.
    
    This function first checks for local checkpoints (for backward compatibility),
    then falls back to downloading from Hugging Face.
    
    Arguments
    ---------
    model_type: Type of model checkpoint to retrieve
    local_data_dir: Directory containing local checkpoints (optional)
    cache_dir: Directory to cache downloaded checkpoints (optional)
    force_download: Whether to force download even if local file exists
        
    Returns
    -------
    Path to the checkpoint file
    """
    manager = CheckpointManager(cache_dir=cache_dir)

    # Check for local checkpoint first
    if local_data_dir and not force_download:
        checkpoint_info = MODEL_CHECKPOINTS[model_type]
        local_path = os.path.join(local_data_dir, checkpoint_info['filename'])

        if manager.check_local_checkpoint(model_type, local_path):
            print(f"Using local checkpoint: {local_path}")
            return local_path
        else:
            print(f"Local checkpoint not found or invalid: {local_path}")
            print(f"Downloading from Hugging Face: {checkpoint_info['filename']} or loading from cache.")

    return manager.get_checkpoint_path(model_type, force_download=force_download)
