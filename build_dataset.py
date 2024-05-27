import h5py
import os
import logging
import numpy as np
import gc
import multiprocessing as mp

#saving datas to each part_folders in h5 type
def transform_h5_build_basedata(feature_encoder, ddf, ddf_path, filename, preprocess=False, block_size=0):
    def _transform_block(feature_encoder, df_block, df_block_path, filename, preprocess):
        if preprocess:
            df_block = feature_encoder.preprocess(df_block)
        darray_dict = feature_encoder.transform(df_block)
        save_h5(darray_dict, os.path.join(df_block_path, filename))


    if block_size > 0: 
        block_id = 1
        for idx in range(0, len(ddf), block_size):
            if block_id in list(range(1,4)):
                if block_id in list(range(2,4)):
                    idx = idx + block_size*(block_id-1)
                df_block = ddf[idx: (idx + block_size*2)]
                df_part_path = os.path.join(ddf_path,'part{}'.format(block_id))
                _transform_block(feature_encoder, df_block, df_part_path, filename + '.h5', preprocess) 
                
            if block_id in list(range(4,11)):
                idx = idx + block_size*(block_id-1)
                if block_id == 4:
                    df_block = ddf[idx: (idx + block_size*2)]
                    df_part_path = os.path.join(ddf_path,'part{}'.format(block_id))
                    _transform_block(feature_encoder, df_block, df_part_path, filename + '.h5', preprocess)
                df_block_1 = ddf[idx: (idx + block_size)]
                df_block_2 = ddf[(idx + block_size): (idx + block_size*2)]
                df_part_path = os.path.join(ddf_path,'part{}'.format(block_id))
                _transform_block(feature_encoder, df_block_1, df_part_path, filename + '_1.h5', preprocess) 
                _transform_block(feature_encoder, df_block_2, df_part_path, filename + '_2.h5', preprocess)
            block_id += 1

            if block_id > 10:
                break
                
    else:
        _transform_block(feature_encoder, ddf, ddf_path, filename + ".h5", preprocess)

#saving sample datas to each part_folders in h5 type
def transform_h5_build_interdata(feature_encoder, ddf, ddf_path, filename, preprocess=False, block_size=0):
    def _transform_block(feature_encoder, df_block, df_block_path, filename, preprocess):
        if preprocess:
            df_block = feature_encoder.preprocess(df_block)
        darray_dict = feature_encoder.transform(df_block)
        save_h5(darray_dict, os.path.join(df_block_path, filename))


    if block_size > 0: 
        block_id = 1
        for idx in range(0, len(ddf), block_size):
            df_block = ddf[idx: (idx + block_size)]
            df_part_path = ddf_path
            _transform_block(feature_encoder, df_block, df_part_path, filename + '_{}.h5'.format(block_id), preprocess)        
            block_id += 1

    else:
        _transform_block(feature_encoder, ddf, ddf_path, filename + ".h5", preprocess)
