# *ShEPhERD* checkpoints for PyTorch 2.5.1
### Sept. 3, 2025
Checkpoints have been moved to HuggingFace to reduce repo size. They can be automatically downloaded or manually from here: [https://huggingface.co/kabeywar/shepherd](https://huggingface.co/kabeywar/shepherd).

These checkpoints were converted from the original model weights trained using PyTorch Lightning v1.2 to v2.5.1 using `python -m pytorch_lightning.utilities.upgrade_checkpoint <chkpt_path>`. The original model weights can be found at:
[https://www.dropbox.com/scl/fo/rgn33g9kwthnjt27bsc3m/ADGt-CplyEXSU7u5MKc0aTo?rlkey=fhi74vkktpoj1irl84ehnw95h&e=1&st=wn46d6o2&dl=0](https://www.dropbox.com/scl/fo/rgn33g9kwthnjt27bsc3m/ADGt-CplyEXSU7u5MKc0aTo?rlkey=fhi74vkktpoj1irl84ehnw95h&e=1&st=wn46d6o2&dl=0).

## Available Models

| Model Type | Description | Training Dataset |
|------------|-------------|------------------|
| `mosesaq` | Shape, electrostatics, and pharmacophores | MOSES-aq |
| `gdb_x2` | Shape conditioning only | GDB17 |
| `gdb_x3` | Shape and electrostatics | GDB17 |
| `gdb_x4` | Pharmacophores only | GDB17 |


### Basic Usage

```python
from shepherd import load_shepherd_model

# Load the default MOSES-aq model (downloads automatically if needed)
model = load_shepherd_model()

# Load a specific model type
model = load_shepherd_model('gdb_x3')
```

### Advanced Usage
```python
from shepherd import load_model, clear_model_cache

# Use custom cache directory
model = load_model(cache_dir='./data/shepherd_chkpts')

# Check for local checkpoints first
model = load_model(local_data_dir='./data/shepherd_chkpts')

# Clear cached models
clear_model_cache('mosesaq')  # Clear specific model
clear_model_cache()  # Clear all models
```

### Get Model Information
```python
from shepherd import get_model_info

models = get_model_info()
for model_type, description in models.items():
    print(f"{model_type}: {description}")
```