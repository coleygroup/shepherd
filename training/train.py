import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

import rdkit
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch_geometric
from torch_geometric.nn import radius_graph
import torch_scatter

import pickle
from copy import deepcopy
import os
import shutil
import datetime
import multiprocessing
from tqdm import tqdm

import sys
sys.path.insert(-1, "model/")
sys.path.insert(-1, "model/equiformer_v2")

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from torch_geometric.data import HeteroData

from model.model import Model
from lightning_module import LightningModule
from datasets import HeteroDataset

import importlib

sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)
def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)

if __name__ == '__main__':
    """
    This repository includes only a small subset of the training data so that the repository is self-contained.
    After downloading the full training datasets (see README), change the corresponding lines of code below.
    """
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    
    pl.utilities.seed.seed_everything(seed = args.seed, workers = True)
    
    params = importlib.import_module(f'parameters.{args.model_name}').params
    
    # CHANGE ME ONCE FULL DATASETS ARE DOWNLOADED
    if params['data'] == 'GDB17':
        # sample data
        molblocks_and_charges = []
        with open(f'conformers/gdb/example_molblock_charges.pkl', 'rb') as f:
            molblocks_and_charges = pickle.load(f) 
        """
        # full dataset
        molblocks_and_charges = []
        for i in [0,1,2]:
            with open(f'conformers/gdb/molblock_charges_{i}.pkl', 'rb') as f:
                molblocks_and_charges_ = pickle.load(f) 
            molblocks_and_charges += molblocks_and_charges_
        
        # removing randomly-chosen test-set molecules prior to training
        test_indices = np.load('conformers/gdb/random_split_test_indices.npy')
        for index in tqdm(sorted(test_indices)[::-1]): # removing from end of list
            if index < len(molblocks_and_charges):
                molblocks_and_charges.pop(index)
        """
    
    # CHANGE ME ONCE FULL DATASETS ARE DOWNLOADED
    if params['data'] == 'MOSES_aq':
        # sample data
        molblocks_and_charges = []
        with open(f'conformers/moses_aq/example_molblock_charges.pkl', 'rb') as f:
            molblocks_and_charges = pickle.load(f)    
        """
        # full dataset
        molblocks_and_charges = []
        for i in [0,1,2,3,4]:
            with open(f'conformers/moses_aq/molblock_charges_{i}.pkl', 'rb') as f:
                molblocks_and_charges_ = pickle.load(f) 
            molblocks_and_charges += molblocks_and_charges_
        """
    
    
    dataset = HeteroDataset(
        molblocks_and_charges = molblocks_and_charges, 
        
        noise_schedule_dict = params['noise_schedules'],
        
        explicit_hydrogens = params['dataset']['explicit_hydrogens'],
        use_MMFF94_charges = params['dataset']['use_MMFF94_charges'],
        
        formal_charge_diffusion = params['x1_formal_charge_diffusion'],

        x1 = params['dataset']['compute_x1'],
        x2 = params['dataset']['compute_x2'],
        x3 = params['dataset']['compute_x3'],
        x4 = params['dataset']['compute_x4'],
        
        recenter_x1 = params['dataset']['x1']['recenter'], 
        add_virtual_node_x1 = params['dataset']['x1']['add_virtual_node'],
        remove_noise_COM_x1 = params['dataset']['x1']['remove_noise_COM'],
        atom_types_x1 = params['dataset']['x1']['atom_types'],
        charge_types_x1 = params['dataset']['x1']['charge_types'],
        bond_types_x1 = params['dataset']['x1']['bond_types'],
        scale_atom_features_x1 = params['dataset']['x1']['scale_atom_features'],
        scale_bond_features_x1 = params['dataset']['x1']['scale_bond_features'],

        independent_timesteps_x2 = params['dataset']['x2']['independent_timesteps'],
        recenter_x2 = params['dataset']['x2']['recenter'],
        add_virtual_node_x2 = params['dataset']['x2']['add_virtual_node'],
        remove_noise_COM_x2 = params['dataset']['x2']['remove_noise_COM'],
        num_points_x2 = params['dataset']['x2']['num_points'],
        
        independent_timesteps_x3 = params['dataset']['x3']['independent_timesteps'],
        recenter_x3 = params['dataset']['x3']['recenter'],
        add_virtual_node_x3 = params['dataset']['x3']['add_virtual_node'],
        remove_noise_COM_x3 = params['dataset']['x3']['remove_noise_COM'],
        num_points_x3 = params['dataset']['x3']['num_points'],
        scale_node_features_x3 = params['dataset']['x3']['scale_node_features'],        
        
        independent_timesteps_x4 = params['dataset']['x4']['independent_timesteps'],
        recenter_x4 = params['dataset']['x4']['recenter'],
        add_virtual_node_x4 = params['dataset']['x4']['add_virtual_node'],
        remove_noise_COM_x4 = params['dataset']['x4']['remove_noise_COM'],
        max_node_types_x4 = params['dataset']['x4']['max_node_types'],
        scale_node_features_x4 = params['dataset']['x4']['scale_node_features'],
        scale_vector_features_x4 = params['dataset']['x4']['scale_vector_features'],
        multivectors = params['dataset']['x4']['multivectors'],
        check_accessibility = params['dataset']['x4']['check_accessibility'],
        
        probe_radius = params['dataset']['probe_radius'], # for x2 and x3
        
    )
    
    
    if params['training']['multiprocessing_spawn']:
        train_loader = torch_geometric.loader.DataLoader(
            dataset = dataset,
            num_workers = params['training']['num_workers'],
            batch_size = params['training']['batch_size'],
            shuffle = True,
            multiprocessing_context = multiprocessing.get_context("spawn"),
            worker_init_fn=set_worker_sharing_strategy,
        )
    else:
        train_loader = torch_geometric.loader.DataLoader(
            dataset = dataset,
            num_workers = params['training']['num_workers'],
            batch_size = params['training']['batch_size'],
            shuffle = True,
            worker_init_fn=set_worker_sharing_strategy,
        )
    
    
    output_dir = f"jobs/{params['training']['output_dir']}"
    try: os.mkdir(f"jobs/")
    except: pass
    try: os.mkdir(output_dir)
    except: pass
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k = 0,
        save_last = True,
        monitor="train_loss",
        mode="min",
        dirpath = output_dir,
        filename="best-{step:09d}",
        every_n_train_steps = params['training']['log_every_n_steps'],
    )
    csv_logger = CSVLogger(
        save_dir = output_dir,
        name = 'csv_logger',
    )
    
    gradient_clip_val = params['training']['gradient_clip_val']
    accumulate_grad_batches = params['training']['accumulate_grad_batches']
    
    cuda_available = torch.cuda.is_available()
    from pytorch_lightning.strategies.ddp import DDPStrategy
    trainer = pl.Trainer(
        callbacks = [checkpoint_callback],
        logger = [csv_logger],
        
        default_root_dir = output_dir,
        accelerator = "gpu" if (params['training']['num_gpus'] >= 1 and cuda_available) else 'cpu', 
        
        max_epochs = 10000,
        
        gradient_clip_val = gradient_clip_val,
        accumulate_grad_batches = accumulate_grad_batches,
        
        log_every_n_steps = params['training']['log_every_n_steps'],
        
        reload_dataloaders_every_n_epochs = 1, # re-shuffle training data after each epoch
        
        devices = params['training']['num_gpus'] if cuda_available else "auto",
        
        strategy = DDPStrategy(find_unused_parameters=True) if (params['training']['num_gpus'] > 1 and cuda_available) else None,
        precision = 32,
        
        terminate_on_nan = True,
    )
    
    model_pl = LightningModule(params)
    print(sum(p.numel() for p in model_pl.parameters() if p.requires_grad))
    
    resume_from_checkpoint = True
    ckpt_path = f"{output_dir}/last.ckpt"
    ckpt_path = ckpt_path if (os.path.exists(ckpt_path) & resume_from_checkpoint) else None
    
    # avoid overwriting previous "last.ckpt"
    if (ckpt_path is not None) and (trainer.global_rank == 0):
        date = datetime.datetime.now()
        timestamp = str(date.year) + '_' + str(date.month).zfill(2) + '_' + str(date.day).zfill(2) + '_' + str(date.hour).zfill(2) + '_' + str(date.minute).zfill(2)
        shutil.copyfile(ckpt_path, f"{output_dir}/last_{timestamp}.ckpt")
    
    
    print('beginning to train...')
    trainer.fit(model_pl, train_loader, ckpt_path = ckpt_path)
