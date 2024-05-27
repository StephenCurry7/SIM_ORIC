import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "2"

import gc
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error

import torch
from torch import nn #from tensorflow import keras
from torch.optim import Adam,Adagrad #from tensorflow.python.keras.optimizers import adam_v2,adagrad_v2
from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names

from loaddata import load_basic_data, load_seqence_data, load_feat_info
from CombineModel import Combined_FinalMLP_Model
from oric import ORIC, load_oric

import sys
import logging
import fuxictr_version
from 模块fuxictr import datasets
from datetime import datetime
from 模块fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from 模块fuxictr.features import FeatureMap
from 模块fuxictr.pytorch.torch_utils import seed_everything
from 模块fuxictr.preprocess.处理原始数据_feature_processor import FeatureProcessor
from 模块fuxictr.preprocess.build_dataset import save_h5,transform_h5_build_interdata
from 模块fuxictr.pytorch.dataloaders import H5DataLoader
import src
import gc
import argparse
from pathlib import Path
import h5py
import multiprocessing as mp

torch.cuda.set_device(0)
torch.cuda.init()

class Reservoir(object): #蓄水池，是从当前所有的历史数据中均匀采样出的数量固定的样本，用于与新的数据一起对现有的集成模型进行训练
    def __init__(self, reservoir_size):
        self.reservoir_size = reservoir_size
        self.reservoir = None
        self.count = 0

    def update(self, data):
        if self.reservoir is None:
            self.reservoir = pd.DataFrame(
                np.zeros([self.reservoir_size, len(data.columns)]), #一个shape=[reservoir_size,len(data.columns)]的全为0的np数组，列名称与data的列名称相同
                columns=data.columns)
        
        #一个长度为len(data)的np数组，其中各个数是0(包含)到reservoir_size(不包含)间的随机整数:
        replace = np.random.randint(self.reservoir_size, size=len(data)) 
        keep_length = min(len(data), self.reservoir_size - self.count)
        if keep_length > 0:
            #替换replace的第0个至第keep_length-1个位置为count至count+keep_length-1:
            replace[np.arange(keep_length)] = np.arange(self.count, self.count + keep_length) 

        rand = np.random.rand(len(data)) #生成在[0, 1)区间内均匀分布的随机数组，长度为len(data)
        acc_prob = [self.reservoir_size / (self.count + i) for i in range(1, len(data) + 1)]
        replace = np.where(rand < acc_prob, replace, -1)#replace中各位置满足rand < acc_prob的不变，不满足的替换为-1
        #去除replace重复的元素,并按元素由小到大返回一个新的元组。res_idx为replace中去重且重排序的元素，data_idx为res_idx中各元素在replace[::-1]中的位置
        res_idx, data_idx = np.unique(replace[::-1], return_index=True)
        data_idx = [len(data) - 1 - i for i in data_idx] #res_idx中各元素在replace中的位置
        if res_idx[0] == -1:
            res_idx, data_idx = res_idx[1:], data_idx[1:]

        self.reservoir.iloc[res_idx] = data.iloc[data_idx] #替换reservoir中样本为data中样本，见毕业论文P109
        self.count += len(data)

    def data(self):
        return self.reservoir.iloc[range(min(self.count, self.reservoir_size))]

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

def bulid_feature_names(sparse_feat, dense_feat, nunique_feat, emb_dim, sequence_feat=[], max_len=0): #embedding操作
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
    '''sparse_feat=['userid', 'adgroup_id', 'pid', 'cms_segid', 'cms_group_id', 'final_gender_code', 
                'age_level', 'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level', 'campaign_id', 'customer']'''
    for file_id in train_ids: #train_ids=[1,2,3]
        #在debug时，不用每次都去在数据集1、2、3上预训练ORIC模型，如果之前训练过就跳过预训练环节，从而加速debug速度（oric.fit训练ORIC模型费时间）
        #但是在数据集改变的时候要重新预训练新的ORIC模型,需要把下面的if not os.path.exist部分删掉
        if not os.path.exists('./data/criteo/part{}/oric_nconf{}_decay{}.pkl'.format(file_id,oric_info['n_conf'],oric_info['decay'])):# （我自己加的）
#！！！！！！！！！！！！！！！！！！！注意是否需要重新预训练新的ORIC模型，这里只是为了debug
        # train ORIC
            X, y = load_basic_data(data_folder, file_id)# X为[104858 rows x 14 columns]的dataframe，y为(104858,)的numpy数组
            time_start = time.time()
            oric.fit(X[sparse_feat], y, miss_val) #miss_val=[0]
            add_record([oric_info["n_conf"], oric_info["decay"], file_id],
                        time.time() - time_start,
                        os.path.join(data_folder, "time_oric.pkl")) #存储训练ORIC的时间
    
            # save ORIC
            oric_path = os.path.join(data_folder,
                                         "part{}".format(file_id),
                                         path_with_oric_info("oric", ".pkl", oric_info))
            oric.save(oric_path)

def build_inter_to_h5(inter_input, label, mode, path, oric_info):
    inter_input_columns = []
    for i in inter_input.columns:
        inter_input_columns.append(i) #获取inter_input的列名
        
    inter_input.insert(0, 'label', label) #将相应label的值加入到inter_input中
    
    #修改params中的参数，为后续将inter_input数据转换为h5数据做准备
    params['feature_cols'][0]['name']=[]
    params['feature_cols'][1]['name']=inter_input_columns #交叉特征全为类型型特征
        
    feature_encoder = FeatureProcessor(**params)
    inter_input = feature_encoder.preprocess(inter_input)
    feature_encoder.fit(inter_input,
                        os.path.join(path,'{}_inter_input_feature_encoder_nconf{}_decay{}'.format(mode, oric_info['n_conf'],oric_info['decay'])), 
                        **args) 

    #奇偶情况不一样
    if len(inter_input) % 2 ==0:
        block_size = int(len(inter_input)/2)
    else:
        block_size = int(len(inter_input)/2+1)

    if mode != 'sample':
        transform_h5_build_interdata(feature_encoder, 
                                     inter_input, 
                                     path, 
                                     '{}_inter_data_nconf{}_decay{}'.format(mode, oric_info['n_conf'],oric_info['decay']), 
                                     preprocess=False, 
                                     block_size=block_size) #block_size将inter_input分为train和valid两类，为后续将inter_input分别用于训练、测试做准备
    else:
        transform_h5_build_interdata(feature_encoder, 
                                     inter_input, 
                                     path, 
                                     '{}_inter_data_nconf{}_decay{}'.format(mode, oric_info['n_conf'],oric_info['decay']), 
                                     preprocess=False, 
                                     block_size=0) 

def generate_interactions(file_id, mode, oric):
    file_name = "base_data.pkl" if mode != "sample" else "sample_data.pkl"
    base_input, y = load_basic_data(data_folder, file_id, file_name) #将part_{}.format(file_id)中整理为base_input和_，分别为样本和标签
    inter_input = oric.transform(base_input) #根据oric模型生成交叉特征

    part_folder = os.path.join(data_folder, 'part{}'.format(file_id))
    build_inter_to_h5(inter_input, y, mode, part_folder, oric_info) #将inter_input转换为h5类型，不用再保存csv类型的inter_input

def pretrain_base_model(train_ids, valid_ids): 
    """Pre-train the base model on the available data."""
    #prepare the validation data
    part_folder = os.path.join(data_folder, "part{}".format(valid_ids[0]),) 
    params["train_data"] = os.path.join(part_folder, 'base_data.h5')

    feature_map = FeatureMap(params['dataset_id'], data_folder)
    feature_map_json = os.path.join(data_folder,'feature_map.json')
    feature_map.load(feature_map_json, params)
    #logging.info("Feature specs: " + print_to_json(feature_map.features))

    valid_gen = H5DataLoader(feature_map, stage='train', **params).make_iterator()

    #initialize the base model
    model_class = getattr(src, params['model'])
    base_model = model_class(feature_map, **params)
    base_model.count_parameters()
    
    # #prepare the training data
    for file_id in train_ids:
        if not os.path.exists('./data/criteo/part{}/base_model_{}'.format(file_id,model_type)): #仅为了debug，正式训练时需要删除
            print("now deal with part", file_id)
            part_folder = os.path.join(data_folder, "part{}".format(file_id),) 
            params["train_data"] = os.path.join(part_folder, 'base_data.h5')
            feature_map = FeatureMap(params['dataset_id'], data_folder)
            feature_map_json = os.path.join(data_folder,'feature_map.json')
            feature_map.load(feature_map_json, params)
                
            #prepare the training data
            train_gen = H5DataLoader(feature_map, stage='train', **params).make_iterator()
                
            #pretrain the base model
            base_model_file = os.path.join(part_folder,"base_model_{}".format(model_type)) 
            base_model_weight_file = os.path.join(part_folder,'base_model_weight_{}'.format(model_type))
            base_model.fit(train_gen, validation_data=valid_gen, model_path=base_model_file, weight_path=base_model_weight_file, **params)

def one_step(file_id, oric_info,  evaluate=True, create_interaction=True, #对应论文算法2且fine-tune不使用resevior
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
    #training data
    params["train_data"] = os.path.join(part_folder, 'base_data_1.h5')
    feature_map = FeatureMap(params['dataset_id'], data_folder)
    feature_map_json = os.path.join(data_folder,'feature_map.json')
    feature_map.load(feature_map_json, params)
    
    train_gen = H5DataLoader(feature_map, stage='train', **params).make_iterator()
    #validation data
    params["train_data"] = os.path.join(part_folder, 'base_data_2.h5')
    feature_map = FeatureMap(params['dataset_id'], data_folder)
    feature_map_json = os.path.join(data_folder,'feature_map.json')
    feature_map.load(feature_map_json, params)
    valid_gen = H5DataLoader(feature_map, stage='train', **params).make_iterator()
    
    base_input, y = load_basic_data(data_folder, file_id) #将part_{}.format(file_id)整理为base_input和y
    seq_input = load_seqence_data(data_folder, file_id) #{}

    if create_interaction: #根据原来的ORIC模型和新的数据集生成交叉特征
        # load or initialize ORIC
        previous_oric_path = os.path.join(previous_part_folder,
                                          path_with_oric_info("oric", ".pkl", oric_info))
        if os.path.exists(previous_oric_path): #加载之前训练过的oric模型
            print('exist pretrained oric model on part{}'.format(file_id - 1))
            oric = load_oric(previous_oric_path)
        else:
            print('one_step函数 train oric model on part{}'.format(file_id - 1))
            oric = ORIC(**oric_info)
            
        # create the interactive test data 根据上面的（旧的）ORIC模型生成交叉特征，算法2第1步
        if evaluate: 
            print('one_step函数 用part{}训练好的oric在part{}上生成交叉特征'.format(file_id - 1, file_id))
            #print('one_step函数 用训练好的oric模型生成交叉特征on part{}'.format(file_id - 1))
            if not os.path.exists('./data/criteo/part{}/test_inter_nconf{}_decay{}_1.h5'.format(file_id,oric_info['n_conf'],oric_info['decay'])):
                generate_interactions(file_id, "test", oric) #用旧的ORIC模型在新的数据集上生成交叉特征并保存，用于评估模型
            
        # update ORIC 在新的数据集上训练新的ORIC模型，算法2第4步
        time_start = time.time()
        print('one_step函数在新的数据集part{}上训练oric模型'.format(file_id))
        #！！！！！！！！！！！！！！！！！！！注意是否需要重新预训练新的ORIC模型，这里只是为了debug
        if not os.path.exists('./data/criteo/part{}/oric_nconf{}_decay{}.pkl'.format(file_id,oric_info['n_conf'],oric_info['decay'])): #(我自己加的)
            oric.fit(base_input, y, miss_val)
            oric_running_time = time.time() - time_start
            add_record([oric_info["n_conf"], oric_info["decay"], file_id],
                        oric_running_time,
                        os.path.join(data_folder, "time_oric.pkl"))
            oric_path = os.path.join(part_folder, path_with_oric_info("oric", ".pkl", oric_info))
            oric.save(oric_path)
            print('training time of ORIC:', oric_running_time)
        else: #我自己加的
            oric = load_oric('./data/criteo/part{}/oric_nconf{}_decay{}.pkl'.format(file_id,oric_info['n_conf'],oric_info['decay']))
        
        # create the interactive training data 根据上面的（新的）ORIC模型产生交叉特征，用于训练后续interaction模型和集成模型
        if not os.path.exists('./data/criteo/part{}/train_inter_data_nconf{}_decay{}_1.h5'.format(file_id,oric_info['n_conf'],oric_info['decay'])): #我自己加的
            generate_interactions(file_id, "train", oric) #算法2第5步


    if evaluate: #算法2第2步
        # evaluate the base model 评估base model
        print('evaluate the base_model on part{} using part{}'.format(file_id-1,file_id))
        base_model = torch.load(previous_base_model_file)
        test_metrics = base_model.evaluate([train_gen,valid_gen]) #返回的是一个字典{'AUC':...,'logloss':...}
        for metric, test_metric in test_metrics.items(): #将base_model验证结果保存
            add_record([metric, file_id],
                       test_metric,
                       os.path.join(data_folder, "base_result_{}.pkl".format(model_type))) #model_type=DeepFM

        # evaluate the combined model 评估集成模型
        print('evaluate the combined_model on part{} using part{}'.format(file_id-1,file_id))
        combined_model_file = os.path.join(data_folder, "combine_model_{}".format(model_type))
        combined_model = torch.load(combined_model_file)

        '''test_inter_path = os.path.join(part_folder, path_with_oric_info("test_inter", ".gz", oric_info)) #用于测试的交叉特征的路径
        inter_input = pd.read_csv(test_inter_path) #旧的ORIC在新的数据集上生成的交叉特征'''

        #加载h5类型的inter_input，分别为训练集和验证集
        feature_map_inter_folder = os.path.join(part_folder, 
                                                'test_inter_input_feature_encoder_nconf{}_decay{}'.format(oric_info['n_conf'],oric_info['decay']))
        
        params["train_data"] = os.path.join(part_folder, 'test_inter_data_nconf{}_decay{}_1.h5'.format(oric_info['n_conf'],oric_info['decay']))
        feature_map_inter = FeatureMap(params['dataset_id'], feature_map_inter_folder)
        feature_map_json = os.path.join(feature_map_inter_folder,'feature_map.json')
        feature_map_inter.load(feature_map_json, params)
        #logging.info("Feature specs: " + print_to_json(feature_map.features))
        inter_test_gen_1 = H5DataLoader(feature_map_inter, stage='train', **params).make_iterator()
    
        params["train_data"] = os.path.join(part_folder, 'test_inter_data_nconf{}_decay{}_2.h5'.format(oric_info['n_conf'],oric_info['decay']))
        feature_map_inter = FeatureMap(params['dataset_id'], feature_map_inter_folder)
        feature_map_json = os.path.join(feature_map_inter_folder,'feature_map.json')
        feature_map_inter.load(feature_map_json, params)
        #logging.info("Feature specs: " + print_to_json(feature_map.features))
        inter_test_gen_2 = H5DataLoader(feature_map_inter, stage='train', **params).make_iterator()


        test_metrics = combined_model.combine_evaluate([train_gen,inter_test_gen_1,valid_gen,inter_test_gen_2])

        print('test_metrics for file {}:'.format(file_id),test_metrics )
        for metric, test_metric in test_metrics.items(): #将interact part验证结果保存
            add_record([oric_info["n_conf"], oric_info["decay"], metric, file_id],
                       test_metric,
                       os.path.join(data_folder, "inter_result_{}.pkl".format(model_type)))
            '''add_record([oric_info["n_conf"], oric_info["decay"], metric, file_id],
                       test_metric,
                       os.path.join(data_folder, "inter_result.pkl"))'''

    # fine-tune the base model 用新的数据集更新旧的base_model，算法2第3步
    if train_base_model:
        if not os.path.exists('./data/criteo/part{}/base_model_{}'.format(file_id,model_type)): #仅为了debug，正式训练时需要删除
            print('trainning base model on part{}'.format(file_id))
            base_model = torch.load(previous_base_model_file)
            base_model_file = os.path.join(part_folder, "base_model_{}".format(model_type))
            base_model_weight_file = os.path.join(part_folder,'base_model_weight_{}'.format(model_type))
            print(base_model_file)
        
            time_start = time.time()
            base_model.fit(train_gen, validation_data=valid_gen, model_path=base_model_file, weight_path=base_model_weight_file,
                            **params)#根据新的数据集更新旧的base_model
            base_runing_time = time.time() - time_start
            add_record([file_id],
                        base_runing_time,
                        os.path.join(data_folder, "time_base_model_{}.pkl".format(model_type)))
            print("training time of base model:", base_runing_time)

    if train_combined_model: #训练集成模型，算法2第6步   有问题！训练loss和valid_loss太大！！
        print('trainning combined model on part{}'.format(file_id))
        # load interactions
        '''train_inter_path = os.path.join(part_folder, path_with_oric_info("train_inter", ".gz", oric_info))
        inter_input = pd.read_csv(train_inter_path) #上面if create_interaction中由新的ORIC模型在新的数据集上生成的交叉特征
        
        #将inter_input转换为h5类型
        build_inter_to_h5(inter_input, y, part_folder, oric_info)'''

        #加载h5类型的inter_input，分别为训练集和验证集
        feature_map_inter_folder = os.path.join(part_folder, 
                                                'train_inter_input_feature_encoder_nconf{}_decay{}'.format(oric_info['n_conf'],oric_info['decay']))
        
        params["train_data"] = os.path.join(part_folder, 'train_inter_data_nconf{}_decay{}_1.h5'.format(oric_info['n_conf'],oric_info['decay']))
        feature_map_inter = FeatureMap(params['dataset_id'], feature_map_inter_folder)
        feature_map_json = os.path.join(feature_map_inter_folder,'feature_map.json')
        feature_map_inter.load(feature_map_json, params)
        #logging.info("Feature specs: " + print_to_json(feature_map.features))
        inter_train_gen = H5DataLoader(feature_map_inter, stage='train', **params).make_iterator()
    
        params["train_data"] = os.path.join(part_folder, 'train_inter_data_nconf{}_decay{}_2.h5'.format(oric_info['n_conf'],oric_info['decay']))
        feature_map_inter = FeatureMap(params['dataset_id'], feature_map_inter_folder)
        feature_map_json = os.path.join(feature_map_inter_folder,'feature_map.json')
        feature_map_inter.load(feature_map_json, params)
        #logging.info("Feature specs: " + print_to_json(feature_map.features))
        inter_valid_gen = H5DataLoader(feature_map_inter, stage='train', **params).make_iterator()

        # build the combined model #构造集成模型
        #定义inter part model
        params['use_fs'] = False #inter part不使用特征选择层
        model_class = getattr(src, params['model'])
        inter_model = model_class(feature_map_inter, **params)

        #base part
        if use_previous_model:
            base_model = torch.load(previous_base_model_file) 
            #base_model不用上面更新好的base_model（如下一行），是依照论文4.2最后一段集成模型的训练copy the weights of the previous 集成模型
            #base_model = load_model(base_model_file, custom_objects)
        else:
            params["train_data"] = os.path.join(part_folder, 'base_data.h5')
            feature_map = FeatureMap(params['dataset_id'], data_folder)
            feature_map_json = os.path.join(data_folder,'feature_map.json')
            feature_map.load(feature_map_json, params)
            model_class = getattr(src, params['model'])
            base_model = model_class(feature_map, **params)
        
        combined_model = Combined_FinalMLP_Model(base_model, inter_model, feature_map,output_use_fusion=output_use_fusion, **params)  #集成模型，base_part.trainable = False

        # train the combined model #根据新的数据集训练集成模型(只训练interaction part，因为combine_model函数中base_model.trainable=False)
        combined_model_file = os.path.join(data_folder, "combine_model_{}".format(model_type))
        combined_model_weight_file = os.path.join(data_folder,'combine_model_weight_{}'.format(model_type))

        time_start = time.time()
        
        combined_model.combine_fit([train_gen, inter_train_gen], validation_data=[valid_gen, inter_valid_gen], unfreeze_base_model=False, 
                           model_path=combined_model_file, weight_path=combined_model_weight_file, **params)
        
        combined_running_time = time.time() - time_start

        # fine-tune the combined model （在数据集上微调整个集成模型）
        if fine_tune_combined_model: #True
            print('fine tune combined model on part{}'.format(file_id))
            combined_model = torch.load(combined_model_file)
            combined_model.compile(params["optimizer"], params["loss"], learning_rate_fine_tune) #调整fine_tune的学习率
            #combined_model.trainable = True #使得base part和interaction part
            combined_model.train()
            
            time_start = time.time()
            combined_model.combine_fit([train_gen, inter_train_gen], validation_data=[valid_gen, inter_valid_gen], unfreeze_base_model=True, 
                           model_path=combined_model_file, weight_path=combined_model_weight_file, **params)
            
            combined_running_time += time.time() - time_start

        add_record([oric_info["n_conf"], oric_info["decay"], file_id],
                   combined_running_time,
                   os.path.join(data_folder, "time_combined_model.pkl"))
        print("training time of combined model:", combined_running_time)

    gc.collect()

def one_step_sample(file_id, oric_info, evaluate=True, create_interaction=True, #论文算法2且fine-tune使用reservoir
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
    #training data
    params["train_data"] = os.path.join(part_folder, 'base_data_1.h5')
    feature_map = FeatureMap(params['dataset_id'], data_folder)
    feature_map_json = os.path.join(data_folder,'feature_map.json')
    feature_map.load(feature_map_json, params)
    
    train_gen = H5DataLoader(feature_map, stage='train', **params).make_iterator()
    #validation data
    params["train_data"] = os.path.join(part_folder, 'base_data_2.h5')
    feature_map = FeatureMap(params['dataset_id'], data_folder)
    feature_map_json = os.path.join(data_folder,'feature_map.json')
    feature_map.load(feature_map_json, params)
    
    valid_gen = H5DataLoader(feature_map, stage='train', **params).make_iterator()
    #training sample data
    params["train_data"] = os.path.join(part_folder, 'sample_data.h5')
    feature_map = FeatureMap(params['dataset_id'], part_folder)
    feature_map_json = os.path.join(part_folder,'feature_map.json')
    feature_map.load(feature_map_json, params)
    
    train_sample_gen = H5DataLoader(feature_map, stage='train', **params).make_iterator()

    if create_interaction:
        # load ORIC and create interactions for training
        oric_path = os.path.join(part_folder, path_with_oric_info("oric", ".pkl", oric_info))
        oric = load_oric(oric_path)
        generate_interactions(file_id, "sample", oric)


    if evaluate:
        # evaluate the base model
        print('evaluate the base_model on part{} using part{}'.format(file_id-1,file_id))
        base_model = torch.load(previous_base_model_file)
        test_metrics = base_model.evaluate([train_gen,valid_gen]) #返回的是一个字典{'AUC':...,'logloss':...}
        for metric, test_metric in test_metrics.items():
            add_record([metric, file_id],
                       test_metric,
                       os.path.join(data_folder, "sample_base_result_{}.pkl".format(model_type)))

        # evaluate the combined model
        print('evaluate the combined_model on part{} using part{} with sample data'.format(file_id-1,file_id))
        combined_model_file = os.path.join(data_folder, "sample_combine_model_{}".format(model_type))
        combined_model = torch.load(combined_model_file)

        #加载h5类型的inter_input，分别为训练集和验证集
        feature_map_inter_folder = os.path.join(part_folder, 
                                                'test_inter_input_feature_encoder_nconf{}_decay{}'.format(oric_info['n_conf'],oric_info['decay']))
        
        params["train_data"] = os.path.join(part_folder, 'test_inter_data_nconf{}_decay{}_1.h5'.format(oric_info['n_conf'],oric_info['decay']))
        feature_map_inter = FeatureMap(params['dataset_id'], feature_map_inter_folder)
        feature_map_json = os.path.join(feature_map_inter_folder,'feature_map.json')
        feature_map_inter.load(feature_map_json, params)
        #logging.info("Feature specs: " + print_to_json(feature_map.features))
        inter_test_gen_1 = H5DataLoader(feature_map_inter, stage='train', **params).make_iterator()
    
        params["train_data"] = os.path.join(part_folder, 'test_inter_data_nconf{}_decay{}_2.h5'.format(oric_info['n_conf'],oric_info['decay']))
        feature_map_inter = FeatureMap(params['dataset_id'], feature_map_inter_folder)
        feature_map_json = os.path.join(feature_map_inter_folder,'feature_map.json')
        feature_map_inter.load(feature_map_json, params)
        #logging.info("Feature specs: " + print_to_json(feature_map.features))
        inter_test_gen_2 = H5DataLoader(feature_map_inter, stage='train', **params).make_iterator()

        
        test_metrics = combined_model.combine_evaluate([train_gen,inter_test_gen_1,valid_gen,inter_test_gen_2])
        print('test_metrics for file {}:'.format(file_id),test_metrics )
        for metric, test_metric in test_metrics.items():
            add_record([oric_info["n_conf"], oric_info["decay"], metric, file_id],
                       test_metric,
                       os.path.join(data_folder, "sample_inter_result_{}.pkl".format(model_type)))
            '''add_record([oric_info["n_conf"], oric_info["decay"], metric, file_id],
                       test_metric,
                       os.path.join(data_folder, "sample_inter_result.pkl"))'''


    # fine-tune the base model
    if train_base_model:
        print('trainning base model on part{} with sample data'.format(file_id))
        base_model = torch.load(previous_base_model_file)
        base_model_file = os.path.join(part_folder, "sample_base_model_{}".format(model_type))
        base_model_weight_file = os.path.join(part_folder,'sample_base_model_weight_{}'.format(model_type))
        print(base_model_file)

        time_start = time.time()
        base_model.fit(train_gen, validation_data=valid_gen, model_path=base_model_file, weight_path=base_model_weight_file,
                        **params)#根据新的数据集更新旧的base_model
        base_model.fit(train_sample_gen, validation_data=valid_gen, model_path=base_model_file, weight_path=base_model_weight_file,
                        **params)#根据新的数据集更新旧的base_model
        base_running_time = time.time() - time_start
        add_record([file_id],
                   base_running_time,
                   os.path.join(data_folder, "time_sample_base_model_{}.pkl".format(model_type)))
        print("trainning time of base model with sample data:", base_running_time)

    if train_combined_model:
        print('trainning combined model on part{} with sample data'.format(file_id))
        # load interactions
        #加载h5类型的inter_input，分别为训练集和验证集
        feature_map_inter_folder = os.path.join(part_folder, 
                                                'train_inter_input_feature_encoder_nconf{}_decay{}'.format(oric_info['n_conf'],oric_info['decay']))
        
        params["train_data"] = os.path.join(part_folder, 'train_inter_data_nconf{}_decay{}_1.h5'.format(oric_info['n_conf'],oric_info['decay']))
        feature_map_inter = FeatureMap(params['dataset_id'], feature_map_inter_folder)
        feature_map_json = os.path.join(feature_map_inter_folder,'feature_map.json')
        feature_map_inter.load(feature_map_json, params)
        #logging.info("Feature specs: " + print_to_json(feature_map.features))
        inter_train_gen = H5DataLoader(feature_map_inter, stage='train', **params).make_iterator()
    
        params["train_data"] = os.path.join(part_folder, 'train_inter_data_nconf{}_decay{}_2.h5'.format(oric_info['n_conf'],oric_info['decay']))
        feature_map_inter = FeatureMap(params['dataset_id'], feature_map_inter_folder)
        feature_map_json = os.path.join(feature_map_inter_folder,'feature_map.json')
        feature_map_inter.load(feature_map_json, params)
        #logging.info("Feature specs: " + print_to_json(feature_map.features))
        inter_valid_gen = H5DataLoader(feature_map_inter, stage='train', **params).make_iterator()

        # build the combined model #根据inter_valid_gen的params构造集成模型
        #定义inter part model
        params['use_fs'] = False #inter part不使用特征选择层
        model_class = getattr(src, params['model'])
        inter_model = model_class(feature_map_inter, **params)
        
        #加载h5类型的inter_sample_data
        feature_map_inter_sample_folder = os.path.join(part_folder, 
                                                'sample_inter_input_feature_encoder_nconf{}_decay{}'.format(oric_info['n_conf'],oric_info['decay']))
        
        params["train_data"] = os.path.join(part_folder, 'sample_inter_data_nconf{}_decay{}.h5'.format(oric_info['n_conf'],oric_info['decay']))
        feature_map_inter_sample = FeatureMap(params['dataset_id'], feature_map_inter_sample_folder)
        feature_map_json = os.path.join(feature_map_inter_sample_folder,'feature_map.json')
        feature_map_inter_sample.load(feature_map_json, params)
        #logging.info("Feature specs: " + print_to_json(feature_map.features))
        inter_train_sample_gen = H5DataLoader(feature_map_inter_sample, stage='train', **params).make_iterator()

        # build the combined model
        if use_previous_model:
            base_model = torch.load(previous_base_model_file)
        else:
            params["train_data"] = os.path.join(part_folder, 'base_data.h5')
            feature_map = FeatureMap(params['dataset_id'], data_folder)
            feature_map_json = os.path.join(data_folder,'feature_map.json')
            feature_map.load(feature_map_json, params)
            model_class = getattr(src, params['model'])
            base_model = model_class(feature_map, **params)

        combined_model = Combined_FinalMLP_Model(base_model, inter_model, feature_map, output_use_fusion=output_use_fusion, **params)  #集成模型，base_part.trainable = False

        # train the combined model
        combined_model_file = os.path.join(data_folder, "sample_combine_model_{}".format(model_type))
        combined_model_weight_file = os.path.join(data_folder,'sample_combine_model_weight_{}'.format(model_type))

        time_start = time.time()
        
        combined_model.combine_fit([train_gen, inter_train_gen], validation_data=[valid_gen, inter_valid_gen], unfreeze_base_model=False, 
                           model_path=combined_model_file, weight_path=combined_model_weight_file, **params)
        combined_model.combine_fit([train_sample_gen, inter_train_sample_gen], validation_data=[valid_gen, inter_valid_gen], unfreeze_base_model=False, 
                           model_path=combined_model_file, weight_path=combined_model_weight_file, **params)
        
        combined_running_time = time.time() - time_start

        # fine-tune the combined model
        if fine_tune_combined_model:
            print('fine tune combined model on part{} with sample data'.format(file_id))
            combined_model = torch.load(combined_model_file)
            combined_model.train()
            combined_model.compile(params["optimizer"], params["loss"], learning_rate_fine_tune) #调整fine_tune的学习率
            
            time_start = time.time()
            combined_model.combine_fit([train_gen, inter_train_gen], validation_data=[valid_gen, inter_valid_gen], unfreeze_base_model=True, 
                           model_path=combined_model_file, weight_path=combined_model_weight_file, **params)
            combined_model.combine_fit([train_sample_gen, inter_train_sample_gen], validation_data=[valid_gen, inter_valid_gen], unfreeze_base_model=True,
                           model_path=combined_model_file, weight_path=combined_model_weight_file, **params)
            
            combined_running_time += time.time() - time_start

        add_record([oric_info["n_conf"], oric_info["decay"], file_id],
                   combined_running_time,
                   os.path.join(data_folder, "time_sample_combined_model.pkl"))
        print("trainning time of combined model with sample data:", combined_running_time)

    gc.collect()

def select_parameter(para_name, para_vals, train_ids, valid_ids,
                     create_interaction=True, train_base_model=True):
    if train_base_model and len(train_ids) > 1:
        #根据train_ids数据集来训练一个base_model
        pretrain_base_model(train_ids[:-1], train_ids[-1:])    

    for para in para_vals: #para_vals=[10,20,...,100]当选择n_conf时
        print("start: {}={}".format(para_name, para))
        oric_info[para_name] = para #修改oric_info字典中para_name的值
        if create_interaction and len(train_ids) > 1:
            pretrain_oric(train_ids[:-1], oric_info) #根据train_ids预训练一个ORIC模型
        one_step(train_ids[-1], oric_info, False, create_interaction, train_base_model) #根据train_id[-1]训练集成模型一次，依据论文算法2
        for file_id in valid_ids:
            one_step(file_id, oric_info, True, create_interaction, train_base_model) #根据file_id训练并验证集成模型，得到base和interact部分的度量，为后续选择参数做准备
        train_base_model = False #由于base model的训练不受para的影响，因此在第一个para上训练了base model后，后续对其他para就不用再训练base model，直接加载之前训练好的就可以
        print('---------------------------------------------------------------')
    
    print('begin to choose the best {} !!!' .format(para_name))
    best_para, best_loss = 0, float("inf")
    for para in para_vals:
        oric_info[para_name] = para
        valid_loss = read_loss(valid_ids[0], "inter", oric_info) #combined model在验证集上的logloss验证结果
        if valid_loss < best_loss:
            best_para, best_loss = para, valid_loss
    
    print('best_para {}={}'.format(para_name,best_para))
    print('best_loss {}={}'.format(metrics[0],best_loss))
    return best_para

def fine_tune(test_ids, create_interaction=True, train_base_model=True,
              train_combined_model=True, use_previous_model=True):
    args = (create_interaction, train_base_model, train_combined_model, use_previous_model)
    one_step(test_ids[0] - 1, oric_info, False, *args) #用part5数据集和选好的最优参数更新base model和ORIC模型，并更新集成模型
    for file_id in test_ids: #test_ids=[6,7,8,9,10]
        one_step(file_id, oric_info, True, *args)

def sampled_retrain(test_ids, create_interaction=True, train_base_model=True,
                    train_combined_model=True, use_previous_model=True):
    args = (create_interaction, train_base_model, train_combined_model, use_previous_model)
    one_step_sample(test_ids[0] - 1, oric_info, False, *args) #在part5数据上用原数据+reservior的数据训练combined model
    for file_id in test_ids:
        one_step_sample(file_id, oric_info, True, *args)

if __name__ == "__main__":
        torch.set_num_threads(2)
        os.environ["OMP_NUM_THREADS"] = "2"
        random_state = 2024
        device = 'cuda:0'

        dataset = "criteo"
        model_type = "FinalMLP"  # xDeepFM, DeepFM, WDL, DCN, DualMLP, FinalMLP
        data_folder = os.path.join("data", dataset)
        task = "binary"  # "regression" 在params中已有，根据FinalMLP的即可

        oric_info = {
            "n_freq": 100,
            "n_conf": 50,  
            "max_size": 4,
            "n_chain": 10000,
            "decay": 1, #0.4, 
            "online": True,
            "positive_class": True, 
        }
        miss_val = [0]
        
        #根据论文P8 6.1.3进行选择：
        train_ids = list(range(1, 5)) #[1, 2, 3, 4]
        valid_ids = list(range(5, 6)) #[5]
        test_ids = list(range(6, 11)) #[6, 7, 8, 9, 10]
        reservoir_size = 2000000 #蓄水池大小
        n_splits = 2
        n_valid = 1

        #epochs = 30 在params中已有，根据FinalMLP的即可
        batch_size = 8192
        #patience = 0 在params中已有early_stop_patience，根据FinalMLP的即可
        learning_rate_fine_tune = 1e-5#1e-3 对于taobao数据集此学习率为0，见论文6.1.4节和表2
        if learning_rate_fine_tune > 0:
            fine_tune_combined_model = True #若fine_tune的学习率为0，则表示不用再重修训练base model
        else:
            fine_tune_combined_model = False
        patience_ft = 3
        emb_dim = 16
        #loss = "binary_crossentropy" if task == "binary" else "mse"  在params中已有，根据FinalMLP的即可
        metrics = ["logloss", "auc"] if task == "binary" else ["mse"]  #在params中已有，根据FinalMLP的即可
        opt = "adam"  # adam, adagrad

        output_use_fusion = True

        #DualMLP/FinalMLP模型参数设置
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
        parser.add_argument('--expid', type=str, default='DeepFM_test', help='The experiment id to run.')
        parser.add_argument('--gpu', type=int, default=0, help='The gpu index, -1 for cpu')

        args = vars(parser.parse_args(args=[]))
        args['config'] =  './config/FinalMLP_criteo_x1' #'./config/DualMLP_criteo_x1'
        args['expid'] =  'FinalMLP_criteo_x1_004_d5d36917' #'DualMLP_criteo_x1_001_0aa31de8'
        
        experiment_id = args['expid']
        params = load_config(args['config'], experiment_id)
        params['gpu'] = args['gpu'] #包含模型DualMLP参数信息的列表
        params['batch_size'] = batch_size
        set_logger(params)
        #logging.info("Params: " + print_to_json(params))
        #seed_everything(seed=random_state)  #设置随机种子，可以复现结果。在训练时暂时不需要

        # pretrain the models and select the parameters 
        #在数据集part1-4上训练更新base part和ORIC模型，并在part4上训练interact part和集成模型，并根据算法2在part5上验证和更新集成模型，并选择参数
        n_confs = range(10, 101, 10) #list(n_confs)=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        oric_info["n_conf"] = select_parameter("n_conf", n_confs, train_ids, valid_ids)
        decays = list([n/10 for n in range(11)]) 
        oric_info["decay"] = select_parameter("decay", decays, train_ids, valid_ids, train_base_model=False) 
        #因为base_model已经在选择n_conf时都训练好了，且不受n_conf影响，所以不再训练base_model；但是oric模型受n_conf影响，所以还要训练，基于最好的n_conf训练
        
        print(oric_info)
        
        # fine-tune the models on the test data
        #先用part5和最优参数更新base model和ORIC，并得到集成模型；再用part6-10和最优参数不断更新base model、ORIC和集成模型，且评估集成模型和base model
        fine_tune(test_ids, True, True, True) 

        # fine-tune the models with reservoir  
        #create_reservoir() 已经在run process criteo文件中运行
        #先用part5数据+reservior数据训练集成模型；再用part6-10+reservior数据不断更新集成模型，且用part6-10数据评估base model和集成模型
        sampled_retrain(test_ids, True, False, True)