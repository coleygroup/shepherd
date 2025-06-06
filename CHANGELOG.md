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