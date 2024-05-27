#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import gc
from sklearn.metrics import log_loss, mean_squared_error

'''import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import  save_model,load_model,clone_model
from tensorflow.python.keras import Model
from tensorflow.keras import layers
from deepctr.models import xDeepFM, DeepFM, WDL, DCN, AutoInt'''
import torch
from torch import nn #from tensorflow import keras
from torch.nn import Module #from tensorflow.python.keras import Model
from deepctr_torch.models import xDeepFM, DeepFM, WDL, DCN, AutoInt #from deepctr.models import xDeepFM, DeepFM, WDL, DCN, AutoInt
from deepctr_torch.models.basemodel import BaseModel
import copy #from tensorflow.python.keras.models import clone_model
from deepctr_torch.inputs import combined_dnn_input
import numpy as np

def build_model(model_type, linear_feature_columns, dnn_feature_columns, task="binary"):
    if model_type == "DeepFM":
        model = DeepFM(linear_feature_columns,
           dnn_feature_columns,
           task=task,
           dnn_hidden_units=[400, 400, 400],
           device='cuda:0',
        )
    elif model_type == "xDeepFM":
        model = xDeepFM(
            linear_feature_columns,
            dnn_feature_columns,
            task=task,
            dnn_hidden_units=[400, 400],
            cin_layer_size=[200, 200, 200],
            device='cuda:0',
        )
    elif model_type == "WDL":
        model = WDL(
            linear_feature_columns,
            dnn_feature_columns,
            task=task,
            dnn_hidden_units=[1024, 512, 256],
            device='cuda:0',
        )
    elif model_type == "DCN":
        model = DCN(
            linear_feature_columns,
            dnn_feature_columns,
            task=task,
            dnn_hidden_units=[1024, 1024],
            cross_num=6,
            device='cuda:0',
        )
    else:
        model = AutoInt(
            linear_feature_columns,
            dnn_feature_columns,
            task=task,
            dnn_hidden_units=[400, 400],
            att_embedding_size=64,
            device='cuda:0'
        )
    return model







# def combine_model(model_type,  #use_previous_model=False时候的combine_model函数
#         base_linear_feat,
#         base_dnn_feat,
#         inter_linear_feat,
#         inter_dnn_feat):
#     base_part = build_model(model_type, base_linear_feat, base_dnn_feat)
#     base_part._name = "base_part"
#     for layer in base_part.layers:
#         layer._name = "base_" + layer.name
#     base_input = base_part.input
#     base_output = base_part.get_layer(base_part.layers[-2].name).output
#
#     inter_part = build_model(model_type, inter_linear_feat, inter_dnn_feat)
#     inter_part._name = "inter_part"
#     for layer in inter_part.layers:
#         layer._name = "inter_" + layer.name
#     inter_input = inter_part.input
#     inter_output = inter_part.get_layer(inter_part.layers[-2].name).output
#
#     x = layers.add([base_output, inter_output])
#     x = tf.sigmoid(x)
#     final_output = tf.reshape(x, (-1, 1))
#     model = Model(
#         inputs=[base_input, inter_input],
#         outputs=final_output,
#     )
#     return model


def train_model(model,
                train_data,
                valid_data,
                epochs,
                batch_size,
                patience,
                model_checkpoint_file,
                task,):
    best_valid_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs): #epochs=30
        breakout = False
        for file_id, (input_batch, y_batch) in enumerate(train_data):  #file_id=0,1,2,3
            print("epoch", epoch, "file", file_id)

            if len(input_batch) == 2:#此时input_batch为一个长为2的字典{'base_input':[,...,],'inter_input':[,...,]}
                base_input_batch = input_batch['base_input'] #一个与feature_nums长度相同的列表，列表各元素对应一个base特征构成的dataframe，尺寸为batch_size
                inter_input_batch = input_batch['inter_input'] #一个与inter_feature_nums长度相同的列表，列表各元素对应一个inter特征构成的dataframe，尺寸为batch_size
                input_batch = np.concatenate((base_input_batch,inter_input_batch)) #尺寸为(feature_nums+inter_feature_nums,batch_size)
                input_batch=list(input_batch)#长度为feature_nums+inter_feature_nums的列表，列表各元素对应一个base特征构成的numpy，尺寸为(batch_size)
                model.cuda()
            
            model.fit(input_batch,
                      y_batch,
                      shuffle=True,
                      batch_size=batch_size, #batch_size=8192
                      epochs=1,
                      verbose=2,
                      )

            valid_loss = 0
            for input_valid, y_valid in valid_data:
                if len(input_valid) == 2:
                    base_input_valid = input_valid['base_input'] #一个与feature_nums长度相同的列表，列表各元素对应一个base特征构成的dataframe，尺寸为batch_size
                    inter_input_valid = input_valid['inter_input'] #一个与inter_feature_nums长度相同的列表，列表各元素对应一个inter特征构成的dataframe，尺寸为batch_size
                    input_valid = np.concatenate((base_input_valid,inter_input_valid)) #尺寸为(feature_nums+inter_feature_nums,batch_size)
                    input_valid=list(input_valid)
                
                pred_valid = model.predict(input_valid, batch_size=batch_size)
                if task == "binary":
                    valid_loss += log_loss(y_valid, pred_valid, eps=1e-7)
                else:
                    valid_loss += mean_squared_error(y_valid, pred_valid)
            valid_loss /= len(valid_data)
            print('valid_logloss:', valid_loss) #print('valid_loss:', valid_loss)

            if valid_loss < best_valid_loss:
                #model.save(model_checkpoint_file) 
                torch.save(model,model_checkpoint_file) 
                print(
                    "[%d-%d] model saved!. Valid loss improved from %.4f to %.4f"
                    % (epoch, file_id, best_valid_loss, valid_loss)
                )
                best_valid_loss = valid_loss
                patience_counter = 0
            else:
                if patience_counter >= patience:
                    breakout = True
                    print("Early Stopping!")
                    break
                patience_counter += 1
            gc.collect() #垃圾回收

        if breakout:
            break

