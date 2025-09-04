# September 3, 2025 (v0.2.4)
### Model loading and repository optimization
- Added automatic model downloading from HuggingFace Hub with `load_model()`, `get_model_info()`, and `clear_model_cache()` functions
- Removed model weights from git history to reduce repository size - **users should re-clone the repository**
- Added ability for interrupting inference with improved UI to Streamlit app

# August 29, 2025 (v0.2.3)
### Add Streamlit app for demonstrations
- Added an easy-to-use app for demonstration purposes

# August 29, 2025 (v0.2.2)
### Refresh inference code
- Refactored ShEPhERD inference code to be more modular (backwards compatible).
    - The original inference code can still be imported: `from shepherd.inference import inference_sample`
    - New inference functions can be imported with:
        - `from shepherd.inference import generate`
        - `from shepherd.inference import generate_from_intermediate_time`
- Inference now supports atom and bond inpainting
    - `generate` is updated to allow atom inpainting from t=T
    - `generate_from_intermediate_time` is specialized to allow atom inpainting from an intermediate time (T ≤ t < 0)
- Inference can store full diffusion trajectories by setting `return_trajectories=True` during sampling.
- `shepherd.extract`
    - Added `remove_side_groups_with_geometry`
    - Added `remove_overlaps` to quickly filter sampled molecules that use atom-inpainting.


# June 5, 2025 (v0.2.0)
### Refactoring and upgrades for PyTorch >= v2.5.1

- Refactored ShEPhERD (aided by Matthew Cox's fork: https://github.com/mcox3406/shepherd/)
    - Updated import statements: throughout repo to import directly from `shepherd` assuming local install.
    - Fix depreciation warnings:
        - `torch.load()` -> `torch.load(weights_only=True)`
        - `@torch.cuda.amp.autocast(enabled=False)` -> `@torch.amp.autocast('cuda', enabled=False)`
    - Training scripts
        - Updated `src/shepherd/datasets.py` for higher versions of PyG. Required changes to the batching functionality for edges (still backwards compatible).
        - Slight changes to `training/train.py` for upgraded versions of PyTorch Lightning.
- Model checkpoints have been UPDATED for PyTorch Lightning v2.5.1
    - The original checkpoints for PyTorch Lightning v1.2 can be found in previous commits (`c3d5ec0` or before), the original publication Release, or at the Dropbox data link: https://www.dropbox.com/scl/fo/rgn33g9kwthnjt27bsc3m/ADGt-CplyEXSU7u5MKc0aTo?rlkey=fhi74vkktpoj1irl84ehnw95h&e=1&st=wn46d6o2&dl=0
- Created a basic unconditional generation test script
- Updated the environment and relevant files to be compatible with PyTorch >= v2.5.1
- Bug fix for `shepherd.datasets.HeteroDataset.__getitem__` where x3 point extraction should use `get_x2_data`

#### Additional notes
Thank you to Matthew Cox for his contributions in the updated code.


# January 13, 2025
- Added the ability to do partial inpainting for pharmacophores at inference.