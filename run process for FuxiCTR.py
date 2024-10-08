import os
import sys
import logging
import fuxictr_version
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.preprocess.feature_processor import FeatureProcessor
from fuxictr.preprocess.build_dataset import save_h5,transform_h5
from build_dataste import transform_h5_build_basedata
from fuxictr.pytorch.dataloaders import H5DataLoader
import src
import gc
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import h5py
import multiprocessing as mp

class Reservoir(object): 
    def __init__(self, reservoir_size):
        self.reservoir_size = reservoir_size
        self.reservoir = None
        self.count = 0

    def update(self, data):
        if self.reservoir is None:
            self.reservoir = pd.DataFrame(
                np.zeros([self.reservoir_size, len(data.columns)]), 
                columns=data.columns)
        
        replace = np.random.randint(self.reservoir_size, size=len(data)) 
        keep_length = min(len(data), self.reservoir_size - self.count)
        if keep_length > 0:
            replace[np.arange(keep_length)] = np.arange(self.count, self.count + keep_length) 

        rand = np.random.rand(len(data)) 
        acc_prob = [self.reservoir_size / (self.count + i) for i in range(1, len(data) + 1)]
        replace = np.where(rand < acc_prob, replace, -1)
        res_idx, data_idx = np.unique(replace[::-1], return_index=True)
        data_idx = [len(data) - 1 - i for i in data_idx]
        if res_idx[0] == -1:
            res_idx, data_idx = res_idx[1:], data_idx[1:]

        self.reservoir.iloc[res_idx] = data.iloc[data_idx]
        self.count += len(data)

    def data(self):
        return self.reservoir.iloc[range(min(self.count, self.reservoir_size))]

##model parameter setting
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
parser.add_argument('--expid', type=str, default='DeepFM_test', help='The experiment id to run.')
parser.add_argument('--gpu', type=int, default=0, help='The gpu index, -1 for cpu')

args = vars(parser.parse_args(args=[]))
args['config'] = './config/FinalMLP_criteo_x1' #'./config/DualMLP_criteo_x1'
args['expid'] = 'FinalMLP_criteo_x1_004_d5d36917' #'DualMLP_criteo_x1_001_0aa31de8'

experiment_id = args['expid']
params = load_config(args['config'], experiment_id)
params['gpu'] = args['gpu'] 
set_logger(params)
logging.info("Params: " + print_to_json(params))
seed_everything(seed=params['seed'])

##Convert the raw data to the h5 type required for FuxiCTR and store it in each part folder
dataset = 'criteo' #'avazu','taobao'
data_folder = os.path.join('./data', dataset)
base_data_path = os.path.join(data_folder, 'train_clean.txt')

logging.info("Reading file: " + base_data_path)
base_data_df = pd.read_csv(base_data_path, memory_map=True) 
base_data_df.rename(columns={'click':'label'},inplace=True)
feature_encoder = FeatureProcessor(**params)
base_data_df = feature_encoder.preprocess(base_data_df) 
feature_encoder.fit(base_data_df, data_folder, **args)
transform_h5_build_basedata(feature_encoder, base_data_df, data_folder, 'base_data', preprocess=False, block_size=int(len(base_data_df)/20+1)) 



##Pre-processed base_data is sampled with reservior, then the sampled data in each part is converted into h5 files required by FuxiCTR
reservoir = Reservoir(reservoir_size=2000000)
for file_id in list(range(1,11)):
    part_folder = os.path.join(data_folder, "part{}".format(file_id))
    data = pd.read_pickle(os.path.join(part_folder, "base_data.pkl"))
    reservoir.update(data)
    sample_path = os.path.join(part_folder, "sample_data.pkl")
    reservoir.data().to_pickle(sample_path) 

    sample_df = pd.read_pickle(sample_path)
    sample_df.rename(columns={'click':'label'},inplace=True)
    feature_encoder_sample =FeatureProcessor(**params)
    sample_df = feature_encoder_sample.preprocess(sample_df)
    feature_encoder_sample.fit(sample_df, part_folder, **args)
    transform_h5(feature_encoder_sample, sample_df, part_folder, 'sample_data', preprocess=False, block_size=0)
    print("save reservoir for part {} done".format(file_id))