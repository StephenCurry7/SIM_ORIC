import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "2"

import gc
import time
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error

import torch
from torch import nn 
from torch.optim import Adam,Adagrad 
from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names

from loaddata import load_basic_data, load_seqence_data, load_feat_info
from Combine_Model_FinalMLP import Combined_FinalMLP_Model, Combined_DualMLP_Model
from FinalMLP import FinalMLP
from DualMLP import DualMLP
from oric import ORIC, load_oric
from trainmodel_pytorch import build_model, train_model

import sys
import logging
import fuxictr_version
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.preprocess.feature_processor import FeatureProcessor
from uxictr.preprocess.build_dataset import save_h5,transform_h5_build_interdata
from fuxictr.pytorch.dataloaders import H5DataLoader
import src
import gc
import argparse
from pathlib import Path
import h5py
import multiprocessing as mp

from time_sequence_generate import data_collection, frequency_sequence_generate, conf_sequence_generate
from time_sequence_model import net_training

torch.cuda.set_device(0)
torch.cuda.init()

def path_with_oric_info(prefix, postfix, oric_info):
    return prefix + "_nconf{}_decay{}".format(oric_info["n_conf"], oric_info["decay"]) + postfix

def load_record(fp):
    if os.path.exists(fp):
        with open(fp, "rb") as handle:
            record = pickle.load(handle)
        return record
    else:
        raise ValueError("file does not exist!")

def add_record(key_list, val, fp):
    record = load_record(fp) if os.path.exists(fp) else {}
    tmp_dict = record
    print(record)
    for key in key_list[:-1]:
        if key not in tmp_dict:
            tmp_dict[key] = {}
        tmp_dict = tmp_dict[key]
    tmp_dict[key_list[-1]] = val
    with open(fp, "wb") as handle:
        pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_loss(file_id, type, oric_info={}):
    if type == "base":
        result_path = os.path.join(data_folder, "base_result_{}.pkl".format(model_type))
        return load_record(result_path)[metrics[0]][file_id]
    else:
        result_path = os.path.join(data_folder, "inter_result_{}.pkl".format(model_type))
        return load_record(result_path)[oric_info["n_conf"]][oric_info["decay"]][metrics[0]][file_id]

def bulid_feature_names(sparse_feat, dense_feat, nunique_feat, emb_dim, sequence_feat=[], max_len=0): 
    fixlen_feature_columns = \
        [SparseFeat(feat,
                    vocabulary_size=nunique_feat[feat] + 1,
                    embedding_dim=emb_dim)
         for feat in sparse_feat] + \
        [DenseFeat(feat, 1, ) for feat in dense_feat]
    
    varlen_feature_columns = \
        [VarLenSparseFeat(
            SparseFeat(feat,
                       vocabulary_size=nunique_feat[feat] + 1,
                       embedding_dim=emb_dim),
            maxlen=max_len,
            combiner='mean',
            weight_name=None)
            for feat in sequence_feat]
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    return dnn_feature_columns, linear_feature_columns, feature_names

def df_to_input(data, feature_names, seq_input={}, idx=None):
    if idx is not None:
        model_input = []
        for name in feature_names:
            if name not in seq_input:
                model_input.append(data[name].iloc[idx])
            else:
                model_input.append(seq_input[name][idx])
    else:
        model_input = []
        for name in feature_names:
            if name not in seq_input:
                model_input.append(data[name])
            else:
                model_input.append(seq_input[name])
    return model_input

def load_base_data(file_ids, feature_names):
    data = []
    for file_id in file_ids:
        X, y = load_basic_data(data_folder, file_id)
        seq_input = load_seqence_data(data_folder, file_id)
        data.append([df_to_input(X, feature_names, seq_input), y])
    return data

def pretrain_oric(train_ids, oric_info):
    """Pre-train ORIC on the available data."""
    # initialize ORIC
    oric = ORIC(**oric_info)

    feat_info_path = os.path.join(data_folder, "feat_info.pkl")
    _, _, sparse_feat, _, _ = load_feat_info(feat_info_path)

    for file_id in train_ids: 
        # train ORIC
        X, y = load_basic_data(data_folder, file_id)
        time_start = time.time()
        oric.fit(X[sparse_feat], y, miss_val)
        add_record([oric_info["n_conf"], oric_info["decay"], file_id],
                    time.time() - time_start,
                    os.path.join(data_folder, "time_oric.pkl"))
    
        # save ORIC
        oric_path = os.path.join(data_folder,
                                "part{}".format(file_id),
                                path_with_oric_info("oric", ".pkl", oric_info))
        oric.save(oric_path)
        
def generate_interactions(file_id, mode, oric):
    file_name = "base_data.pkl" if mode != "sample" else "sample_data.pkl"
    base_input, _ = load_basic_data(data_folder, file_id, file_name)
    inter_input = oric.transform(base_input) 

    inter_path = os.path.join(data_folder,
                              "part{}".format(file_id),
                              "{}_inter_nconf{}_decay{}.gz".format(mode, oric.n_conf, oric.decay))
    inter_input.to_csv(inter_path, compression='gzip', index=False) 
    
def pretrain_base_model(train_ids, valid_ids): 
    """Pre-train the base model on the available data."""
    # initialize the base model
    feat_info_path = os.path.join(datasource_folder, "feat_info.pkl")
    _, dense_feat, sparse_feat, _, nunique_feat = load_feat_info(feat_info_path)
    
    linear_feature_columns, dnn_feature_columns, feature_names = bulid_feature_names(sparse_feat, dense_feat, nunique_feat, emb_dim) 

    base_model = FinalMLP(sparse_feat, dense_feat, linear_feature_columns, dnn_feature_columns, task, device, **params_model).cuda()

    base_model_file = os.path.join(data_folder,  
                                "part{}".format(train_ids[-1]),
                                "base_model_{}".format(model_type)) 
    train_data = load_base_data(train_ids, sparse_feat + dense_feat)
    valid_data = load_base_data(valid_ids, sparse_feat + dense_feat)
    base_model.compile(opt, learning_rate, loss, metrics=[], )
    train_model(base_model,
                train_data,
                valid_data,
                epochs,
                batch_size,
                patience,
                base_model_file,
                task)

def create_reservoir():
    reservoir = Reservoir(reservoir_size)
    for file_id in train_ids + test_ids:
        partsource_folder = os.path.join(datasource_folder, "part{}".format(file_id))
        data = pd.read_pickle(os.path.join(partsource_folder, "base_data.pkl"))
        reservoir.update(data)
        part_folder = os.path.join(data_folder, "part{}".format(file_id))
        sample_path = os.path.join(part_folder, "sample_data.pkl")
        reservoir.data().to_pickle(sample_path)
        print("save reservoir for part {} done".format(file_id))

def evaluate_model(model, X, y, metrics): 
    if len(X) == 2: 
        X0 = X[0]
        X1 = X[1]
        X = np.concatenate((X0,X1))
        X = list(X)
    pred = model.predict(X, batch_size=batch_size) 
    test_metrics = []
    for metric in metrics:
        if metric == "logloss":
            test_metrics.append(round(log_loss(y, pred, eps=1e-7), 7))
        elif metric == "auc":
            test_metrics.append(round(roc_auc_score(y, pred), 7))
        elif metric == "mse":
            test_metrics.append(round(mean_squared_error(y, pred), 7))
    return test_metrics
    

def one_step(file_id, oric_info,  evaluate=True, create_interaction=True, 
    train_base_model=True, train_combined_model=True, use_previous_model=True):
    """Update the base model, combined model with new data.
    If evaluate==True, evaluate the models before update.
    If create_interaction==True, update ORIC and generate the interactions.
    If train_base_model==True, update the base model.
    If train_combined_model==True, train the combined model.
    """

    print("now deal with part", file_id)
    part_folder = os.path.join(data_folder, "part{}".format(file_id), )
    previous_part_folder = os.path.join(data_folder, "part{}".format(file_id - 1), )
    previous_base_model_file = os.path.join(previous_part_folder, "base_model_{}".format(model_type))

    # load data
    base_input, y = load_basic_data(datasource_folder, file_id) 
    seq_input = load_seqence_data(datasource_folder, file_id) 
    feat_info_path = os.path.join(data_folder, "feat_info.pkl")
    _, _, sparse_feat, _, _ = load_feat_info(feat_info_path)

    if create_interaction: 
        # load or initialize ORIC
        previous_oric_path = os.path.join(previous_part_folder,
                                          path_with_oric_info("oric", ".pkl", oric_info))
        if os.path.exists(previous_oric_path): 
            print('exist pretrained oric model on part{}'.format(file_id - 1))
            oric = load_oric(previous_oric_path)
        else:
            oric = ORIC(**oric_info)
            
        # create the interactive test data 
        if evaluate: 
            generate_interactions(file_id, "test", oric) 
            
        # update ORIC 
        time_start = time.time()
        oric.fit(base_input[sparse_feat], y, miss_val)
        oric_running_time = time.time() - time_start
        add_record([oric_info["n_conf"], oric_info["decay"], file_id],
                    oric_running_time,
                    os.path.join(data_folder, "time_oric.pkl"))
        oric_path = os.path.join(part_folder, path_with_oric_info("oric", ".pkl", oric_info))
        oric.save(oric_path)
        print('training time of ORIC:', oric_running_time)
        
        # create the interactive training data 
        generate_interactions(file_id, "train", oric) 


    if evaluate: 
        # evaluate the base model 
        print('evaluate the base_model on part{} using part{}'.format(file_id-1,file_id))
        base_model = torch.load(previous_base_model_file)
        test_metrics = base_model.evaluate([train_gen,valid_gen]) 
        print('test_metrics for file {} on base model:'.format(file_id),test_metrics )
        for metric, test_metric in zip(metrics, test_metrics): 
            add_record([metric, file_id],
                       test_metric,
                       os.path.join(data_folder, "base_result_{}.pkl".format(model_type))) 

        ## evaluate the combined model 
        print('evaluate the combined_model on part{} using part{}'.format(file_id-1,file_id))
        combined_model_file = os.path.join(data_folder, "combine_model_{}".format(model_type))
        combined_model = torch.load(combined_model_file)

        test_inter_path = os.path.join(part_folder, path_with_oric_info("test_inter", ".gz", oric_info)) 
        inter_input = pd.read_csv(test_inter_path) 
        inter_feat = list(inter_input.columns)
        inter_linear_features, inter_dnn_features, inter_feature_names = \
            bulid_feature_names(inter_feat, [], {f: 2 for f in inter_feat}, emb_dim)

        test_metrics = evaluate_model(combined_model,
                                      [df_to_input(base_input, feature_names),
                                       df_to_input(inter_input, inter_feature_names)],
                                      y,  
                                      metrics) 

        print('test_metrics for file {} on combined model:'.format(file_id),test_metrics )
        for metric, test_metric in zip(metrics, test_metrics): 
            add_record([oric_info["n_conf"], oric_info["decay"], metric, file_id],
                       test_metric,
                       os.path.join(data_folder, "inter_result_{}.pkl".format(model_type)))

    # generate datasets for training
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    indices = [idx for _, idx in skf.split(base_input, y)]
    tra_ids, val_ids = [np.concatenate(indices[:-n_valid])], [np.concatenate(indices[-n_valid:])]

    # fine-tune the base model 
    if train_base_model:
        print('trainning base model on part{}'.format(file_id))
        base_model = torch.load(previous_base_model_file)
        base_model_file = os.path.join(part_folder, "base_model_{}".format(model_type))
        
        train_data = [[df_to_input(base_input, feature_names, idx=idx),
                       y[idx]] for idx in tra_ids] 
        valid_data = [[df_to_input(base_input, feature_names, idx=idx),
                        y[idx]] for idx in val_ids] 
        base_model.compile(opt, learning_rate, loss, metrics=[], )
        
        time_start = time.time()
        train_model(base_model,
                    train_data,
                    valid_data,
                    epochs,
                    batch_size,
                    patience,
                    base_model_file,
                    task) 
        base_running_time = time.time() - time_start
        add_record([file_id],
                    base_running_time,
                    os.path.join(data_folder, "time_base_model_{}.pkl".format(model_type)))
        print("trainning time of base model:", base_running_time)
        
    if train_combined_model: 
        print('trainning combined model on part{}'.format(file_id))
        # load interactions
        train_inter_path = os.path.join(part_folder, path_with_oric_info("train_inter", ".gz", oric_info))
        inter_input = pd.read_csv(train_inter_path) 

        #base part
        if use_previous_model:
            base_model = torch.load(previous_base_model_file) 
            '''base_model_file = os.path.join(part_folder, "base_model_{}".format(model_type))
            base_model = torch.load(base_model_file) '''
        else:
            base_model = FinalMLP(sparse_feat, dense_feat, linear_feature_columns, dnn_feature_columns, task, device, **params_model).cuda()
        
        #inter_part
        inter_feat = list(inter_input.columns) 
        inter_linear_features, inter_dnn_features, inter_feature_names = \
            bulid_feature_names(inter_feat, [], {f: 2 for f in inter_feat}, emb_dim)
        inter_sparse_feat = inter_feature_names
        inter_dense_feat = []
        params_model['fs1_context'] = []
        params_model['fs2_context'] = []
        inter_model = FinalMLP(inter_sparse_feat, 
                    inter_dense_feat,
                    inter_linear_features,
                    inter_dnn_features,
                    task=task,
                    device='cuda:0',
                    **params_model)

        feature_names_num = len(feature_names)
        combined_model = Combined_FinalMLP_Model(base_model, inter_model, task, feature_names_num,linear_feature_columns, dnn_feature_columns,output_use_fusion=output_use_fusion)  #集成模型，base_part.trainable = False
        

        # train the combined model 
        combined_model_file = os.path.join(data_folder, "combine_model_{}".format(model_type))

        train_data = [[{"base_input": df_to_input(base_input, feature_names, idx=idx),
                        "inter_input": df_to_input(inter_input, inter_feature_names, idx=idx)},
                       y[idx]] for idx in tra_ids]
        valid_data = [[{"base_input": df_to_input(base_input, feature_names, idx=idx),
                        "inter_input": df_to_input(inter_input, inter_feature_names, idx=idx)},
                       y[idx]] for idx in val_ids]
        combined_model.compile(opt, learning_rate, loss, metrics=[], )
        time_start = time.time()
        
        train_model(combined_model,
                    train_data,
                    valid_data,
                    epochs,
                    batch_size,
                    patience,
                    combined_model_file,
                    task)
        
        combined_running_time = time.time() - time_start

        # fine-tune the combined model 
        if fine_tune_combined_model: 
            print('fine tune combined model on part{}'.format(file_id))
            combined_model = torch.load(combined_model_file)
            combined_model.compile(opt, learning_rate_fine_tune, loss, metrics=[], )
            combined_model.train()
            
            time_start = time.time()
            train_model(combined_model,
                        train_data,
                        valid_data,
                        epochs,
                        batch_size,
                        patience_ft, 
                        combined_model_file,
                        task)
            
            combined_running_time += time.time() - time_start

        add_record([oric_info["n_conf"], oric_info["decay"], file_id],
                   combined_running_time,
                   os.path.join(data_folder, "time_combined_model.pkl"))
        print("training time of combined model:", combined_running_time)

    gc.collect()

def one_step_sample(file_id, oric_info, evaluate=True, create_interaction=True, 
                    train_base_model=True, train_combined_model=True, use_previous_model=True):
    """Update the base model, combined model with new data and the reservoir.
    If evaluate==True, evaluate the models before update.
    If create_interaction==True, update ORIC and generate the interactions.
    If train_base_model==True, update the base model.
    If train_base_model==True, update the base model.
    If train_combined_model==True, train the combined model.
    """

    print("now deal with part", file_id)
    part_folder = os.path.join(data_folder, "part{}".format(file_id), )
    previous_part_folder = os.path.join(data_folder, "part{}".format(file_id - 1), )
    previous_base_model_file = os.path.join(previous_part_folder,
                                            "sample_base_model_{}".format(model_type))
    if not os.path.exists(previous_base_model_file):
        previous_base_model_file = os.path.join(previous_part_folder,
                                                "base_model_{}".format(model_type))

    # load data
    base_input, y = load_basic_data(datasource_folder, file_id)
    sample_input, sample_y = load_basic_data(datasource_folder, file_id, file_name="sample_data.pkl")

    if create_interaction:
        # load ORIC and create interactions for training
        oric_path = os.path.join(part_folder, path_with_oric_info("oric", ".pkl", oric_info))
        oric = load_oric(oric_path)
        generate_interactions(file_id, "sample", oric)

    # load feature columns
    feat_info_path = os.path.join(datasource_folder, "feat_info.pkl")
    _, dense_feat, sparse_feat, _, nunique_feat = load_feat_info(feat_info_path)
    linear_feature_columns, dnn_feature_columns, feature_names = \
        bulid_feature_names(sparse_feat, dense_feat, nunique_feat, emb_dim)


    if evaluate:
        # evaluate the base model
        print('evaluate the base_model on part{} using part{}'.format(file_id-1,file_id))
        base_model = torch.load(previous_base_model_file)
        test_metrics = evaluate_model(base_model,
                                      df_to_input(base_input, feature_names),
                                      y,
                                      metrics)
        for metric, test_metric in zip(metrics, test_metrics):
            add_record([metric, file_id],
                       test_metric,
                       os.path.join(data_folder, "sample_base_result_{}.pkl".format(model_type)))

        # evaluate the combined model
        print('evaluate the combined_model on part{} using part{} with sample data'.format(file_id-1,file_id))
        combined_model_file = os.path.join(data_folder, "sample_combine_model_{}".format(model_type))
        combined_model = torch.load(combined_model_file)

        test_inter_path = os.path.join(part_folder, path_with_oric_info("test_inter", ".gz", oric_info))
        inter_input = pd.read_csv(test_inter_path)
        inter_feat = list(inter_input.columns)
        inter_linear_features, inter_dnn_features, inter_feature_names = \
            bulid_feature_names(inter_feat, [], {f: 2 for f in inter_feat}, emb_dim)

        
        test_metrics = evaluate_model(combined_model,
                                      [df_to_input(base_input, feature_names),
                                       df_to_input(inter_input, inter_feature_names)],
                                      y,
                                      metrics)
        print('test_metrics for file {}:'.format(file_id),test_metrics )
        for metric, test_metric in zip(metrics, test_metrics):
            add_record([oric_info["n_conf"], oric_info["decay"], metric, file_id],
                       test_metric,
                       os.path.join(data_folder, "sample_inter_result_{}.pkl".format(model_type)))
            
    # generate datasets for training
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    indices = [idx for _, idx in skf.split(base_input, y)]
    tra_ids, val_ids = [np.concatenate(indices[:-n_valid])], [np.concatenate(indices[-n_valid:])]

    # fine-tune the base model
    if train_base_model:
        print('trainning base model on part{} with sample data'.format(file_id))
        base_model = torch.load(previous_base_model_file)
        base_model_file = os.path.join(part_folder, "sample_base_model_{}".format(model_type))

        time_start = time.time()
        train_data = [[df_to_input(
            pd.concat([base_input.iloc[tra_ids[0]], sample_input], axis=0),
            feature_names),
            np.concatenate([y[tra_ids[0]], sample_y], axis=0)]]
        valid_data = [[df_to_input(base_input, feature_names, idx=idx), y[idx]]
                      for idx in val_ids]
        
        base_model.compile(opt, learning_rate, loss, metrics=[], )
        time_start = time.time()
        train_model(base_model,
                    train_data,
                    valid_data,
                    epochs,
                    batch_size,
                    patience,
                    base_model_file,
                    task)
        base_running_time = time.time() - time_start
        add_record([file_id],
                   base_running_time,
                   os.path.join(data_folder, "time_sample_base_model_{}.pkl".format(model_type)))
        print("trainning time of base model with sample data:", base_running_time)

    if train_combined_model:
        print('trainning combined model on part{} with sample data'.format(file_id))

        # build the combined model
        if use_previous_model:
            base_model = torch.load(previous_base_model_file)
            '''base_model_file = os.path.join(part_folder, "base_model_{}".format(model_type))
            base_model = torch.load(base_model_file) '''
        else:
            base_model = FinalMLP(sparse_feat, dense_feat, linear_feature_columns, dnn_feature_columns, task, device, **params_model).cuda()

        # load interactions
        train_inter_path = os.path.join(part_folder, path_with_oric_info("train_inter", ".gz", oric_info))
        inter_input = pd.read_csv(train_inter_path)
        sample_inter_path = os.path.join(part_folder, path_with_oric_info("sample_inter", ".gz", oric_info))
        sample_inter_input = pd.read_csv(sample_inter_path)

        inter_feat = list(inter_input.columns)
        inter_linear_features, inter_dnn_features, inter_feature_names = \
            bulid_feature_names(inter_feat, [], {f: 2 for f in inter_feat}, emb_dim)
        inter_sparse_feat = inter_feature_names 
        inter_dense_feat = []

        params_model['fs1_context'] = []
        params_model['fs2_context'] = []
        inter_model = FinalMLP(inter_sparse_feat, 
                    inter_dense_feat,
                    inter_linear_features,
                    inter_dnn_features,
                    task=task,
                    device='cuda:0',
                    **params_model)

        feature_names_num = len(feature_names)
        combined_model = Combined_FinalMLP_Model(base_model, inter_model, task, feature_names_num,linear_feature_columns, dnn_feature_columns,output_use_fusion=output_use_fusion)  #集成模型，base_part.trainable = False
        
        # train the combined model
        combined_model_file = os.path.join(data_folder, "sample_combine_model_{}".format(model_type))

        train_data = [[{
            "base_input": df_to_input(
                pd.concat([base_input.iloc[tra_ids[0]], sample_input], axis=0),
                feature_names),
            "inter_input": df_to_input(
                pd.concat([inter_input.iloc[tra_ids[0]], sample_inter_input], axis=0),
                inter_feature_names),
        },
            np.concatenate([y[tra_ids[0]], sample_y], axis=0)]]
        valid_data = [[{"base_input": df_to_input(base_input, feature_names, idx=idx),
                        "inter_input": df_to_input(inter_input, inter_feature_names, idx=idx)},
                       y[idx]] for idx in val_ids]
        combined_model.compile(opt, learning_rate, loss, metrics=[], )
        time_start = time.time()
        train_model(combined_model,
                    train_data,
                    valid_data,
                    epochs,
                    batch_size,
                    patience,
                    combined_model_file,
                    task)
        combined_running_time = time.time() - time_start

        # fine-tune the combined model
        if fine_tune_combined_model:
            print('fine tune combined model on part{} with sample data'.format(file_id))
            combined_model = torch.load(combined_model_file)
            combined_model.train()
            combined_model.compile(opt, learning_rate_fine_tune, loss, metrics=[], ) 
            
            time_start = time.time()
            train_model(combined_model,
                        train_data,
                        valid_data,
                        epochs,
                        batch_size,
                        patience_ft,
                        combined_model_file,
                        task)
            
            combined_running_time += time.time() - time_start

        add_record([oric_info["n_conf"], oric_info["decay"], file_id],
                   combined_running_time,
                   os.path.join(data_folder, "time_sample_combined_model.pkl"))
        print("trainning time of combined model with sample data:", combined_running_time)

    gc.collect()

def select_parameter(para_name, para_vals, train_ids, valid_ids,
                     create_interaction=True, train_base_model=True):
    if train_base_model and len(train_ids) > 1:
        pretrain_base_model(train_ids[:-1], train_ids[-1:])    

    for para in para_vals: 
        print("start: {}={}".format(para_name, para))
        oric_info[para_name] = para 
        if create_interaction and len(train_ids) > 1:
            pretrain_oric(train_ids[:-1], oric_info) 
        one_step(train_ids[-1], oric_info, False, create_interaction, train_base_model) 
        for file_id in valid_ids:
            one_step(file_id, oric_info, True, create_interaction, train_base_model) 
        train_base_model = False
        print('---------------------------------------------------------------')
    
    print('begin to choose the best {} !!!' .format(para_name))
    best_para, best_loss = 0, float("inf")
    for para in para_vals:
        oric_info[para_name] = para
        valid_loss = read_loss(valid_ids[0], "inter", oric_info) 
        if valid_loss < best_loss:
            best_para, best_loss = para, valid_loss
    '''best_para, best_auc = 0, 0
    for para in para_vals:
        oric_info[para_name] = para
        valid_auc = read_loss(valid_ids[0], "inter", oric_info) 
        if valid_auc > best_auc:
            best_para, best_auc = para, valid_auc'''
    
    print('best_para {}={}'.format(para_name,best_para))
    print('best_loss {}={}'.format(metrics[0],best_loss))
    #print('best_auc {}={}'.format(metrics[1],best_auc))
    return best_para

def fine_tune(test_ids, create_interaction=True, train_base_model=True,
              train_combined_model=True, use_previous_model=True):
    args = (create_interaction, train_base_model, train_combined_model, use_previous_model)
    one_step(test_ids[0] - 1, oric_info, False, *args) 
    for file_id in test_ids: 
        one_step(file_id, oric_info, True, *args)

def sampled_retrain(test_ids, create_interaction=True, train_base_model=True,
                    train_combined_model=True, use_previous_model=True):
    args = (create_interaction, train_base_model, train_combined_model, use_previous_model)
    one_step_sample(test_ids[0] - 1, oric_info, False, *args) 
    for file_id in test_ids:
        one_step_sample(file_id, oric_info, True, *args)

if __name__ == "__main__":
        torch.set_num_threads(2)
        os.environ["OMP_NUM_THREADS"] = "2"
        random_state = 2024
        device = 'cuda:0'

        dataset = "criteo" #'avazu','criteo','taobao'
        model_type = "FinalMLP"  # DualMLP, FinalMLP
        data_folder = os.path.join("data", dataset)
        task = "binary"  # "regression" 

        oric_info = {
            "n_freq": 100,
            "n_conf": 50,  
            "max_size": 4,
            "n_chain": 10000,
            "decay": 1, 
            "online": True,
            "positive_class": True, 
        }
        miss_val = [0]
        
        train_ids = list(range(1, 5)) 
        valid_ids = list(range(5, 6)) 
        test_ids = list(range(6, 11))
        reservoir_size = 2000000 
        n_splits = 2
        n_valid = 1

        batch_size = 8192
        learning_rate_fine_tune = 1e-5
        if learning_rate_fine_tune > 0:
            fine_tune_combined_model = True 
        else:
            fine_tune_combined_model = False
        patience_ft = 3
        emb_dim = 16
        metrics = ["logloss", "auc"] if task == "binary" else ["mse"] 
        opt = "adam"  


        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
        parser.add_argument('--expid', type=str, default='DeepFM_test', help='The experiment id to run.')
        parser.add_argument('--gpu', type=int, default=0, help='The gpu index, -1 for cpu')

        args = vars(parser.parse_args(args=[]))
        args['config'] =  './config/FinalMLP_criteo_x1' #'./config/DualMLP_criteo_x1'
        args['expid'] =  'FinalMLP_criteo_x1_004_d5d36917' #'DualMLP_criteo_x1_001_0aa31de8'
        
        experiment_id = args['expid']
        params = load_config(args['config'], experiment_id)
        params['gpu'] = args['gpu'] 
        params['batch_size'] = batch_size
        set_logger(params)
        #logging.info("Params: " + print_to_json(params))

        params_model = params.copy() #params_model专门用于构建模型

        useless_params = ['learning_rate','task','dataset_id','debug_mode','early_stop_patience','epochs','eval_steps','feature_specs','group_id','loss',
                         'metrics','model','model_id','model_root','monitor','monitor_mode','num_workers','optimizer','ordered_features','pickle_feature_encoder',
                         'save_best_only','shuffle','verbose','data_format','data_root','feature_cols','label_col','min_categr_count','test_data','train_data',
                         'valid_data','gpu']
        for each in useless_params:
            params_model.pop(each)

        # pretrain the models and select the parameters 
        n_confs = range(10, 101, 10) 
        oric_info["n_conf"] = select_parameter("n_conf", n_confs, train_ids, valid_ids)
        decays = list([n/10 for n in range(11)]) 
        oric_info["decay"] = select_parameter("decay", decays, train_ids, valid_ids, train_base_model=False) 
        
        print(oric_info)
        
        # fine-tune the models on the test data
        fine_tune(test_ids, True, True, True) 

        # fine-tune the models with reservoir  
        sampled_retrain(test_ids, True, False, True)
