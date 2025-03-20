#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RadSeg Data Augmentations
ICASSP 2024 - "Multi-Stage Learning for Radar Pulse Activity Segmentation" 
Created on May 7, 2023
@author: Zi Huang
"""

import numpy as np

def subdivideSignalsDataset(dataset: np.ndarray, splits: int=2) -> np.ndarray:
    """ Takes raw I/Q dataset and splits into subsets.
        E.g., shape (100, 128) becomes (200, 64) for a split of 2.
        This is used to create more data in the batch dim. """
    assert len(dataset.shape) == 2, "Shape must be (batch, sequence)."
    assert splits < dataset.shape[-1], "Split must be less than sequence length."
    assert dataset.shape[-1] % splits == 0, "Split must be divisible by sequence length."
    split_dataset = np.split(dataset, int(splits), axis=-1) # Split last axis
    stacked_dataset = np.concatenate(split_dataset, axis=0) # Stack first axis
    return stacked_dataset

def subdivideLabelsDataset(dataset: np.ndarray, splits: int=2) -> np.ndarray:
    """ Takes raw label dataset and splits into subsets.
        E.g., shape (100, 6, 128) becomes (200, 6, 64) for a split of 2.
        This is used to create more data in the batch dim. """
    assert len(dataset.shape) == 3, "Shape must be (batch, channel, sequence)."
    assert splits < dataset.shape[-1], "Split must be less than sequence length."
    assert dataset.shape[-1] % splits == 0, "Split must be divisible by sequence length."
    split_dataset = np.split(dataset, int(splits), axis=-1) # Split last axis
    stacked_dataset = np.concatenate(split_dataset, axis=0) # Stack first axis
    return stacked_dataset

def subdivideSNRsDataset(dataset: np.ndarray, dups: int=2) -> np.ndarray:
    """ Takes raw SNR dataset and splits into subsets.
        E.g., shape (100, 1) becomes (200, 1) for a split of 2.
        SNR is unique, because there is only 1 SNR per original signal, so we need to copy it to match batch dim.
        This is used to create more data in the batch dim. """
    assert len(dataset.shape) == 2, "Shape must be (batch, 1)."
    dup_dataset = np.tile(dataset, (dups, 1)) # Duplicate along vert axis
    stacked_dataset = np.concatenate([dup_dataset], axis=0) # Stack first axis
    return stacked_dataset