import numpy as np

import torch
import torch_geometric
import torch_scatter
from copy import deepcopy

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from model.model import Model

class LightningModule(pl.LightningModule):
    
    def __init__(self, params):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.params = params
        
        self.model = Model(params)
        
        self.train_x1_denoising = params['training']['train_x1_denoising']
        self.train_x2_denoising = params['training']['train_x2_denoising']
        self.train_x3_denoising = params['training']['train_x3_denoising']
        self.train_x4_denoising = params['training']['train_x4_denoising']
        
        self.lr = params['training']['lr']
        self.min_lr = params['training']['min_lr']
        self.lr_steps = params['training']['lr_steps']
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        
        # exponential lr decay from self.lr to self.min_lr in self.lr_steps steps
        gamma = (self.min_lr / self.lr) ** (1.0 / self.lr_steps)
        func = lambda step: max(gamma**(step), self.min_lr / self.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = func)
        
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": False,
            "name": None,
        }
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
    
    def get_training_input_dict(self, data):
        
        input_dict = {}
        
        if self.params['dataset']['compute_x1']:
            input_dict['x1'] = {
                
                # the decoder/denoiser uses the forward-noised structures
                'decoder': {
                    'pos': data['x1'].pos_forward_noised, # this is the structure after forward-noising
                    'x': data['x1'].x_forward_noised, # this is the structure after forward-noising
                    'batch': data['x1'].batch,
                    
                    'bond_edge_mask': data['x1'].bond_edge_mask, # used only for denoising loss calculation
                    'bond_edge_index': data['x1', 'x1'].bond_edge_index, #data['x1'].bond_edge_index,
                    'bond_edge_x': data['x1'].bond_edge_x_forward_noised, # this is the structure after forward-noising
                    
                    'timestep': data['x1'].timestep,
                    'alpha_t': data['x1'].alpha_t,
                    'sigma_t': data['x1'].sigma_t,
                    'alpha_dash_t': data['x1'].alpha_dash_t,
                    'sigma_dash_t': data['x1'].sigma_dash_t,
                    
                    'virtual_node_mask': data['x1'].virtual_node_mask,
                    
                    'pos_noise': data['x1'].pos_noise, # this is the added (gaussian) noise
                    'x_noise': data['x1'].x_noise, # this is the added (gaussian) noise
                    'bond_edge_x_noise': data['x1'].bond_edge_x_noise, # this is the added (gaussian) noise
                    
                },
            }
        
        
        if self.params['dataset']['compute_x2']:
            input_dict['x2'] =  {
                
                # the decoder/denoiser uses the forward-noised structures
                'decoder': {
                    'pos': data['x2'].pos_forward_noised, # this is the structure after forward-noising
                    'x': data['x2'].x_forward_noised, # currently, this is just one-hot embedding of virtual / real node (equal to data['x2'].x)
                    'batch': data['x2'].batch,
                    
                    'timestep': data['x2'].timestep,
                    'alpha_t': data['x2'].alpha_t,
                    'sigma_t': data['x2'].sigma_t,
                    'alpha_dash_t': data['x2'].alpha_dash_t,
                    'sigma_dash_t': data['x2'].sigma_dash_t,
                    
                    'virtual_node_mask': data['x2'].virtual_node_mask,
                    
                    'pos_noise': data['x2'].pos_noise, # this is the added (gaussian) noise
                    
                },
            }
        
        
        if self.params['dataset']['compute_x3']:
            input_dict['x3'] = {
                
                # the decoder/denoiser uses the forward-noised structures
                'decoder': {
                    'pos': data['x3'].pos_forward_noised, # this is the structure after forward-noising
                    'x': data['x3'].x_forward_noised, # this is the structure after forward-noising
                    'batch': data['x3'].batch,
                    
                    'timestep': data['x3'].timestep,
                    'alpha_t': data['x3'].alpha_t,
                    'sigma_t': data['x3'].sigma_t,
                    'alpha_dash_t': data['x3'].alpha_dash_t,
                    'sigma_dash_t': data['x3'].sigma_dash_t,
                    
                    'virtual_node_mask': data['x3'].virtual_node_mask,
                    
                    'pos_noise': data['x3'].pos_noise, # this is the added (gaussian) noise
                    'x_noise': data['x3'].x_noise, # this is the added (gaussian) noise
                    
                },
            }
        
        
        if self.params['dataset']['compute_x4']:
            input_dict['x4'] = {
                
                # the decoder/denoiser uses the forward-noised structures
                'decoder': {
                    'x': data['x4'].x_forward_noised, # this is the structure after forward-noising
                    'pos': data['x4'].pos_forward_noised, # this is the structure after forward-noising
                    'direction': data['x4'].direction_forward_noised, # this is the structure after forward-noising
                    'batch': data['x4'].batch,
                    
                    'timestep': data['x4'].timestep,
                    'alpha_t': data['x4'].alpha_t,
                    'sigma_t': data['x4'].sigma_t,
                    'alpha_dash_t': data['x4'].alpha_dash_t,
                    'sigma_dash_t': data['x4'].sigma_dash_t,
                    
                    'virtual_node_mask': data['x4'].virtual_node_mask,
                    
                    'direction_noise': data['x4'].direction_noise, # this is the added (gaussian) noise
                    'pos_noise': data['x4'].pos_noise, # this is the added (gaussian) noise
                    'x_noise': data['x4'].x_noise, # this is the added (gaussian) noise
                    
                },
            }
        
        input_dict['device'] = self.device
        input_dict['dtype'] = torch.float32
        return input_dict
    
    
    def forward_training(self, input_dict):
        _, output_dict = self.model.forward(input_dict)
        return output_dict
    
    
    def training_step(self, train_batch, batch_idx):
        data = train_batch
        batch_size = data.molecule_id.shape[0]
        
        input_dict = self.get_training_input_dict(data)
        
        output_dict = self.forward_training(input_dict)
        
        loss = 0.0
        #loss = torch.tensor(0.0, requires_grad=True)
        if self.train_x1_denoising:
            loss_x1, feature_loss_x1, pos_loss_x1, bond_loss_x1 = self.x1_denoising_loss(input_dict, output_dict)
            loss = loss + loss_x1
            
            batch_size_nodes = (~input_dict['x1']['decoder']['virtual_node_mask']).sum().item()
            batch_size_edges = input_dict['x1']['decoder']['bond_edge_x_noise'].shape[0]
            
            self.log('train_loss_x1', loss_x1, batch_size = batch_size_nodes)
            self.log('train_pos_loss_x1', pos_loss_x1, batch_size = batch_size_nodes)
            self.log('train_feature_loss_x1', feature_loss_x1, batch_size = batch_size_nodes)
            self.log('train_bond_loss_x1', bond_loss_x1, batch_size = batch_size_edges)
            
        if self.train_x2_denoising:
            loss_x2 = self.x2_denoising_loss(input_dict, output_dict)
            loss = loss + loss_x2
            
            batch_size_nodes = (~input_dict['x2']['decoder']['virtual_node_mask']).sum().item()
            
            self.log('train_loss_x2', loss_x2, batch_size = batch_size_nodes)
            
        if self.train_x3_denoising:
            loss_x3, feature_loss_x3, pos_loss_x3 = self.x3_denoising_loss(input_dict, output_dict)
            loss = loss + loss_x3
            
            batch_size_nodes = (~input_dict['x3']['decoder']['virtual_node_mask']).sum().item()
            
            self.log('train_loss_x3', loss_x3, batch_size = batch_size_nodes)
            self.log('train_pos_loss_x3', pos_loss_x3, batch_size = batch_size_nodes)
            self.log('train_feature_loss_x3', feature_loss_x3, batch_size = batch_size_nodes)
        
        if self.train_x4_denoising:
            loss_x4, feature_loss_x4, pos_loss_x4, direction_loss_x4 = self.x4_denoising_loss(input_dict, output_dict)
            loss = loss + loss_x4
            
            batch_size_nodes = (~input_dict['x4']['decoder']['virtual_node_mask']).sum().item()
            
            self.log('train_loss_x4', loss_x4, batch_size = batch_size_nodes)
            self.log('train_pos_loss_x4', pos_loss_x4, batch_size = batch_size_nodes)
            self.log('train_direction_loss_x4', direction_loss_x4, batch_size = batch_size_nodes)
            self.log('train_feature_loss_x4', feature_loss_x4, batch_size = batch_size_nodes)
        
        self.log('train_loss', loss, batch_size = batch_size)
        return loss
    
    
    def x1_denoising_loss(self, input_dict, output_dict):
        
        mask = ~input_dict['x1']['decoder']['virtual_node_mask']
        pos_loss = torch.mean(
                (input_dict['x1']['decoder']['pos_noise'] - output_dict['x1']['decoder']['denoiser']['pos_out'])[mask] ** 2.0
        )
        
        feature_loss = torch.mean(
            (input_dict['x1']['decoder']['x_noise'] - output_dict['x1']['decoder']['denoiser']['x_out'])[mask] ** 2.0
        )
        
        bond_loss = torch.zeros_like(feature_loss)
        if self.model.x1_bond_diffusion:
            
            input_dict['x1']['decoder']['bond_edge_index']
            
            true_noise = input_dict['x1']['decoder']['bond_edge_x_noise']
            pred_noise = output_dict['x1']['decoder']['denoiser']['bond_edge_x_out']
            bond_mask = input_dict['x1']['decoder']['bond_edge_mask'] # indicates real-bond (True) or non-bond (False)
            
            # weighting contributions from real-bonds and non-bonds equally
                # otherwise, the loss from the non-bonds will overwhelm the loss from the real-bonds
            bond_loss = (torch.mean((true_noise - pred_noise)[bond_mask] ** 2.0) + torch.mean((true_noise - pred_noise)[~bond_mask] ** 2.0)) * 0.5
            
        loss = pos_loss + feature_loss + bond_loss
        
        return loss, feature_loss, pos_loss, bond_loss

    
    def x2_denoising_loss(self, input_dict, output_dict):
        
        mask = ~input_dict['x2']['decoder']['virtual_node_mask']
        pos_loss = torch.mean(
                (input_dict['x2']['decoder']['pos_noise'] - output_dict['x2']['decoder']['denoiser']['pos_out'])[mask] ** 2.0
        )
        
        loss = pos_loss
        
        return loss
    
    
    def x3_denoising_loss(self, input_dict, output_dict):
        
        mask = ~input_dict['x3']['decoder']['virtual_node_mask']
        feature_loss = torch.mean(
            (input_dict['x3']['decoder']['x_noise'] - output_dict['x3']['decoder']['denoiser']['x_out'].squeeze())[mask] ** 2.0
        )
        
        pos_loss = torch.mean(
            (input_dict['x3']['decoder']['pos_noise'] - output_dict['x3']['decoder']['denoiser']['pos_out'])[mask] ** 2.0
        )
        
        loss = feature_loss + pos_loss
        
        return loss, feature_loss, pos_loss
    
    
    def x4_denoising_loss(self, input_dict, output_dict):
        
        mask = ~input_dict['x4']['decoder']['virtual_node_mask']
        if sum(mask) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        feature_loss = torch.mean(
            (input_dict['x4']['decoder']['x_noise'] - output_dict['x4']['decoder']['denoiser']['x_out'].squeeze())[mask] ** 2.0
        )
        
        pos_loss = torch.mean(
                (input_dict['x4']['decoder']['pos_noise'] - output_dict['x4']['decoder']['denoiser']['pos_out'])[mask] ** 2.0
        )
        
        direction_loss = torch.mean(
                (input_dict['x4']['decoder']['direction_noise'] - output_dict['x4']['decoder']['denoiser']['direction_out'])[mask] ** 2.0
        )
        
        loss = feature_loss + pos_loss + direction_loss
        
        return loss, feature_loss, pos_loss, direction_loss
        