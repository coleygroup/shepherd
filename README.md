# ShEPhERD (**S**hape, **E**lectrostatics, and **Ph**armacophores **E**xplicit **R**epresentation **D**iffusion)
This repository contains the code necessary to train and sample from ShEPhERD's diffusion model. Note that ShEPhERD has a sister repository, shepherd-score (https://github.com/coleygroup/shepherd-score), that contains the code for generating/optimizing conformers, extracting interaction profiles, aligning interaction profiles, scoring 3D similarity, and evaluating generated samples from ShEPhERD (validity, 3D similarity, etc.). Both repositories are self-contained and have different installation requirements. Any dependencies on shepherd-score that are necessary to train or sample from ShEPhERD have been copied into `shepherd_score_utils/` for user convenience.

The preprint can be found here: (pending link to arxiv)

<div style="text-align: center;">
  <img src="./shepherd_logo.svg" alt="Shepherd Logo" style="width: 400px; height: auto;">
</div>

## Table of Contents
1. [File Structure](##file-structure)
2. [Environment](##environment)
3. [Training and inference data](##training-and-inference-data)
4. [Training](##training)
5. [Inference](##inference)
6. [Evaluations](##evaluations)

## File Structure

```
.
├── shepherd_score_utils/                       # dependencies from shepherd-score Github repository
├── shepherd_chkpts/                            # trained model checkpoints (from pytorch lightning)
├── paper_experiments/                          # inference scripts for all experiments in preprint
├── samples/                                    # empty dir to hold outputs from paper_experiments/*.py
├── jobs/                                       # empty dir to hold ouputs from train.py
├── conformers/                                 # conditional target structures for experiments, and (sample) training data
├── parameters/                                 # hyperparameter specifications for all models in preprint
├── model/                                      # model architecture
│   │   ├── equiformer_operations.py            # select E3NN operations from (original) Equiformer
│   │   ├── equiformer_v2_encoder.py            # slightly customized Equiformer-V2 module
│   │   └── model.py                            # module definitions and forward passes
│   ├── utils/                                  # misc. functions for forward passes
│   ├── egnn/                                   # customized re-implementation of EGNN
│   └── equiformer_v2/                          # clone of equiformer_v2 codebase, with slight modifications for shepherd
├── train.py                                    # main training script
├── lightning_module.py                         # pytorch-lightning modules and training pipeline
├── datasets.py                                 # torch_geometric dataset class (for training)
├── inference.py                                # inference functions; see Jupyter notebooks for example uses
├── RUNME_conditional_generation_MOSESaq.ipynb  # Jupyter notebook for conditional generation, using MOSES_aq P(x1,x3,x4) model
├── RUNME_unconditional_generation.ipynb        # Jupyter notebook for unconditional generation, for all models
├── environment.yml                             # conda environment requirements
└── README.md
```


## Environment

`environment.yml` contains the conda environment that we used for training and running ShEPhERD. 

**We** followed these steps to create a suitable conda environment, which worked on our Linux system. Please note that this exact installation procedure may depend on your system, particularly your cuda version.

```
conda create --name shepherd python=3.8.13
source activate shepherd
conda install merv::envvar-pythonnousersite-true
source deactivate

source activate shepherd

conda config --append channels conda-forge

pip cache purge
pip3 cache purge
export TMPDIR='/var/tmp'

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit==11.3.1 -c pytorch
conda install pyg=2.2.0 -c pyg

pip install e3nn

pip install jupyterlab

pip install pip==24.0
pip install pytorch-lightning==1.6.3
pip install setuptools==59.5.0

pip install rdkit
conda install xtb
pip install open3d
conda install h5py

pip install numpy --upgrade
```


## Training and inference data
`conformers/` contains the 3D structures of the natural products, PDB ligands, and fragments that we used in our experiments in the preprint. It also includes the 100 test-set structures from GDB-17 that we used in our conditional generation evaluations. 

`conformers/gdb/example_molblock_charges.pkl` contains *sample* training data from our ShEPhERD-GDB-17 training dataset.
`conformers/moses_aq/example_molblock_charges.pkl` contains *sample* training data from our ShEPhERD-MOSES_aq training dataset.

The full training data for both datasets (<10GB each) can be accessed from this Dropbox link: https://www.dropbox.com/scl/fo/rgn33g9kwthnjt27bsc3m/ADGt-CplyEXSU7u5MKc0aTo?rlkey=fhi74vkktpoj1irl84ehnw95h&e=1&st=wn46d6o2&dl=0


## Training
`train.py` is our main training script. It can be run from the command line by specifying a parameter file and a seed. All of our parameter files are held in `parameters/`. As an example, one may re-train the P(x1,x3,x4) model on ShEPhERD-MOSES-aq by calling:

`python train.py params_x1x3x4_diffusion_mosesaq_20240824 0`

Note that the trained checkpoints in `shepherd_chkpts/` were obtained after training each model for ~2 weeks on 2 V100 gpus.


## Inference

The simplest way to run inference is to follow the Jupyter notebooks `RUNME_unconditional_generation_MOSESaq.ipynb` and `RUNME_conditional_generation_MOSESaq.ipynb`. 

`paper_experiments` also contain scripts that we used to run the experiments in our preprint. Each of these scripts should be copied into the parent directory (same directory as this README) before being called from the command line. Some of the scripts (`paper_experiments/run_inference_*_unconditional_*_.py`) take a few additional command-line arguments, which are detailed in those corresponding scripts by argparse commands.


## Evaluations

This repository does *not* contain the code to evaluate samples from ShEPhERD (e.g., evaluate their validity, RMSD upon relaxation, 3D similarity to a target structure, etc). All such evaluations can be found in the sister repository: https://github.com/coleygroup/shepherd-score. These repositories were made separate so that the functions within shepherd-score can be used for more general-purpose applications in ligand-based drug design. We also encourage others to use shepherd-score to evaluate other 3D generative models besides ShEPhERD.


## License

This project is licensed under the MIT License -- see [LICENSE](./LICENSE) file for details.

## Citation
