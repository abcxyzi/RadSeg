#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RadSeg Custom Data Module
ICASSP 2024 - "Multi-Stage Learning for Radar Pulse Activity Segmentation" 
Created on May 7, 2023
@author: Zi Huang
"""

import json 
import numpy as np
import pickle as pk
import pprint 
import torch
from h5 import import_h5
from torch.utils.data import Dataset, DataLoader, random_split
from augmentation import subdivideLabelsDataset, subdivideSignalsDataset, subdivideSNRsDataset

#%% Global
pp = pprint.PrettyPrinter(indent=1, compact=True)
torch.manual_seed(2023)
out_file_labels = "radseg_labels"
out_file_signals = "radseg_iq"
out_file_acts = "radseg_activities"
out_file_acts_meta = "radseg_activities_metadata" # Not used here
out_file_snrs = "radseg_snrs"
out_file_pdws = "radseg_pdws"
out_file_metadata = "radseg_metadata"

#%% RadSeg dataset class
class RadSegDataset(Dataset):
    """ Custom data module for working with the RadSeg dataset. 
        NOTE: Use this module to load the train, val and test sets separately. """
    def __init__(
            self, 
            data_path, # File path
            normalisation_params: dict=None, # Train set statistics, e.g., {"mu": <mean>, "std": <std>}
            is_train: bool=True, # Do normalisation if train data, else use values in normalisation_params for normalisation
            sampler=None, # Used to sample a random interval from the raw I/Q sequence, we may not want to use all the data for training
            sample_window=4096, # Sequence length
            subdivision=None, # Subdivide the dataset into subsets, increases batch dim and descreases seq dim
            transform=None, # Placeholder for other transforms
            enable_mtl: bool=False # RadChar dataset has extra labels for PDWs - this is extra stuff not needed for segmentation
        ):
        # NOTE: Make sure there is enough RAM, this dataset is big
        # NOTE: We only use mask channels 1 to 5 for training, channel 0 is the noise channel - which is not needed for training
        self.loaded_labels = import_h5(dataset_name='labels', file_path=data_path + f'/{out_file_labels}.hdf5')
        self.loaded_signals = import_h5(dataset_name='signals', file_path=data_path + f'/{out_file_signals}.hdf5')
        self.loaded_snrs = import_h5(dataset_name='snrs', file_path=data_path + f'/{out_file_snrs}.hdf5')
        with open(data_path + f"/{out_file_metadata}.json", 'rb') as handle_metadata:
            self.loaded_metadata = json.load(handle_metadata)

        # MTL related
        self.enable_mtl = enable_mtl # NOTE: Not publicly available yet, but hopefully soon...
        if self.enable_mtl:
            # Cannot subdivide if we want to use PDW labels as PDW (e.g., t_delay) no longer makes sense if we are subdividing the I/Q sequence
            if subdivision != None:
                raise ValueError("Cannot use subdivision augmentation if we want to use PDW labels, must set this to None! >:(")
            self.loaded_pdws = import_h5(dataset_name='pdws', file_path=data_path + f'/{out_file_pdws}.hdf5')

        # Subdivision augmentation (use this for training and testing to create more data - this augmentation is used in the ICASSP paper)
        self.transform = transform
        self.subdivision = subdivision
        if self.subdivision is not None:
            self.loaded_labels = subdivideLabelsDataset(dataset=self.loaded_labels, splits=self.subdivision)
            self.loaded_signals = subdivideSignalsDataset(dataset=self.loaded_signals, splits=self.subdivision)
            self.loaded_snrs = subdivideSNRsDataset(dataset=self.loaded_snrs, dups=self.subdivision)

        # Data sampler, to create subset of data
        self.sampler = sampler
        self.sample_window = sample_window # 4096

        # Apply normalisation, bound between -1 and 1
        self.signals_mu, self.signals_var = None, None
        if is_train:
            self.loaded_signals, self.signals_mu, self.signals_var = \
                self.compute_normalisation(self.loaded_signals)
        else:    
            self.signals_mu, self.signals_var = \
                normalisation_params['mu'], normalisation_params['var'] # Mu is a complex number
            self.loaded_signals = \
                self.apply_normalisation(self.loaded_signals, self.signals_mu, self.signals_var)

    def metadata(self) -> dict:
        return self.loaded_metadata
    
    # For signal data, normalise using population mean
    def compute_normalisation(self, signals): # NOTE: Mu is a complex number since we are dealing with complex I/Q signals
        """ Complex number normalisation, Andrew Ng's implementation. """
        mu = np.mean(signals) # Array of [[], [], ...] 
        signals = signals - mu
        var_real = np.var(np.real(signals)) 
        var_imag = np.var(np.imag(signals)) 
        var = var_real + var_imag # Variance is sum of the variances of the real and imaginary parts
        signals = signals/max(var, 1e-12) # Normalise variance
        return signals, mu, var
    
    def apply_normalisation(self, signals, mu, var): # Mu is a complex number
        """ Complex number normalisation. """
        signals = signals - mu # Subtract mean
        signals = signals/max(var, 1e-12) # Normalise variance
        return signals

    def multidim_masks(self, masks):
        """ Takes raw sequence of class labels (loaded_labels), return them as is. """
        return masks # Dim: [mask_channel, sequence_length]

    def random_subinterval_1d(self, array: np.ndarray, interval_size: int) -> np.ndarray:
        """ Returns a random segment from array with a given size. """
        array_length = len(array)
        if interval_size > array_length:
            raise ValueError("Interval size cannot be greater than the array length!")        
        start_index = np.random.randint(0, array_length - interval_size + 1)
        end_index = start_index + interval_size
        sub_interval = array[start_index : end_index]
        return sub_interval, start_index, end_index # Use the same interval for IQ and labels

    def __len__(self) -> int:
        """ Returns size of dataset. """
        return len(self.loaded_signals)
                        
    def __getitem__(self, idx: int):
        """ Returns a single sample by index. """
        signal_real = np.real(self.loaded_signals[idx])
        signal_imag = np.imag(self.loaded_signals[idx])
        snrs = np.array(self.loaded_snrs[idx])

        # PDW label, for MTL only
        if self.enable_mtl:
            pdws = np.array(self.loaded_pdws[idx]) # This is a vector, must flatten this before using it for regression
        
        # Needed for computing metrics
        masks = self.multidim_masks(mask)
        labels = np.array(self.loaded_labels[idx][1:]) # Array of class labels from 1 to 5, we don't need noise channel 0
        
        # Compute random subinterval
        if self.sampler is not None:
            # Must use the same interval
            signal_real, sampled_interval_start, sampled_interval_end = self.random_subinterval_1d(signal_real, self.sample_window)
            signal_imag = signal_imag[sampled_interval_start : sampled_interval_end] # 1D
            masks = masks[:, sampled_interval_start : sampled_interval_end] # 2D 
            labels = labels[:, sampled_interval_start : sampled_interval_end] 

        # NOTE (important): 
        # - masks has shape [batch_size, num_classes, sequence_length], labels has shape [batch_size, sequence_length]
        # - sequence_length dim of labels has values between 0 to num_classes
        # - sequence_length dim of masks has values between 0 to 1
        # - nn.CrossEntropyLoss() requires labels to be the target, and masks (model prediction) to be the input
        if not self.enable_mtl:
            return signal_real, signal_imag, labels, masks, snrs
        else:
            return signal_real, signal_imag, labels, masks, snrs, pdws

#%% Example usage
if __name__ == "__main__":

    # Root
    DATA_PATH = {
        "TRAIN_DATA_PATH": "./RadSeg/train", # NOTE: Replace this as required
        "VAL_DATA_PATH": "./RadSeg/val", 
        "TEST_DATA_PATH": "./RadSeg/test"
    }

    # Configs
    sampler=True, 
    sample_window=int(4096), # Used in ICASSP paper 
    subdivision=2, # Used in ICASSP paper
    enable_mtl=False # NOTE: Not publicly available yet, but hopefully soon...

    # Make train set
    train_set = RadSegDataset(
        data_path=data_path["TRAIN_DATA_PATH"], 
        sampler=sampler, 
        sample_window=sample_window, 
        is_train=True,
        subdivision=subdivision,
        enable_mtl=enable_mtl
    )

    # Compute norm params using train set stats
    train_norm_params = {
        "mu": train_set.signals_mu, # Mu is a complex number 
        "mu_real": np.real(train_set.signals_mu), 
        "mu_imag": np.imag(train_set.signals_mu),
        "var": train_set.signals_var
    }

    # Make validation set and test set
    val_set, test_set = \
        RadSegDataset(
            data_path=data_path["VAL_DATA_PATH"], 
            sampler=sampler, 
            sample_window=sample_window, 
            is_train=False,
            normalisation_params=train_norm_params, 
            subdivision=subdivision,
            enable_mtl=enable_mtl
        ), \
        RadSegDataset(
            data_path=data_path["TEST_DATA_PATH"], 
            sampler=sampler, 
            sample_window=sample_window, 
            is_train=False,          
            normalisation_params=train_norm_params, 
            subdivision=subdivision,
            enable_mtl=enable_mtl
        )