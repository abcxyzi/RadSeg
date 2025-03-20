#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper Methods
ICASSP 2024 - "Multi-Stage Learning for Radar Pulse Activity Segmentation" 
Created on May 7, 2023
@author: Zi Huang
"""

import h5py

def export_h5(data, dataset_name='signals', file_path='./radseg_iq.hdf5'):
    """ Exports dataset to a single HDF5 file. """
    file = h5py.File(file_path, 'w')
    file.create_dataset(dataset_name, data=data)
    file.close()

def import_h5(dataset_name='signals', file_path='./radseg_iq.hdf5'):
    """ Loads a single dataset from a HDF5 file. """
    file = h5py.File(file_path, 'r')
    dataset = file[dataset_name]
    data = dataset[:] # Fetch everything if you have the RAM >:)
    file.close()
    return data