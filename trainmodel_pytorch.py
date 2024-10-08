import os
import gc
from sklearn.metrics import log_loss, mean_squared_error

import torch
from torch import nn 
from torch.nn import Module 
from deepctr_torch.models import xDeepFM, DeepFM, WDL, DCN, AutoInt 
from deepctr_torch.models.basemodel import BaseModel
import copy 
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


def train_model(model,
                train_data,
                valid_data,
                epochs,
                batch_size,
                patience,
                model_checkpoint_file,
                task,):
    best_valid_loss = float("inf") #best_valid_auc = 0
    patience_counter = 0

    for epoch in range(epochs): 
        breakout = False
        for file_id, (input_batch, y_batch) in enumerate(train_data):  
            print("epoch", epoch, "file", file_id)

            if len(input_batch) == 2:
                base_input_batch = input_batch['base_input']
                inter_input_batch = input_batch['inter_input']
                input_batch = np.concatenate((base_input_batch,inter_input_batch)) 
                input_batch=list(input_batch)
                model.cuda()
            
            model.fit(input_batch,
                      y_batch,
                      shuffle=True,
                      batch_size=batch_size,
                      epochs=1,
                      verbose=2,
                      )

            valid_loss = 0
            for input_valid, y_valid in valid_data:
                if len(input_valid) == 2:
                    base_input_valid = input_valid['base_input'] 
                    inter_input_valid = input_valid['inter_input'] 
                    input_valid = np.concatenate((base_input_valid,inter_input_valid))
                    input_valid=list(input_valid)
                
                pred_valid = model.predict(input_valid, batch_size=batch_size)
                if task == "binary":
                    valid_loss += log_loss(y_valid, pred_valid, eps=1e-7)
                else:
                    valid_loss += mean_squared_error(y_valid, pred_valid)
            valid_loss /= len(valid_data)
            print('valid_logloss:', valid_loss) 

            if valid_loss < best_valid_loss:
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
            gc.collect() 

        '''valid_auc = 0
            for input_valid, y_valid in valid_data:
                if len(input_valid) == 2:
                    base_input_valid = input_valid['base_input'] 
                    inter_input_valid = input_valid['inter_input'] 
                    input_valid = np.concatenate((base_input_valid,inter_input_valid))
                    input_valid=list(input_valid)
                
                pred_valid = model.predict(input_valid, batch_size=batch_size)
                if task == "binary":
                    valid_auc += roc_auc_score(y_valid, pred_valid)
                else:
                    valid_auc += mean_squared_error(y_valid, pred_valid)
            valid_auc /= len(valid_data)
            print('valid_auc:', valid_auc)

            if valid_auc > best_valid_auc:
                #model.save(model_checkpoint_file) 
                torch.save(model,model_checkpoint_file,pickle_protocol=4) 
                print(
                    "[%d-%d] model saved!. Valid auc improved from %.4f to %.4f"
                    % (epoch, file_id, best_valid_auc, valid_auc)
                )
                best_valid_auc = valid_auc #best_valid_auc = valid_auc
                patience_counter = 0
            else:
                patience_counter += 1
                print('Stop warning!')
                if patience_counter >= patience:
                    breakout = True
                    print("Early Stopping!")
                    break
            gc.collect() '''

        if breakout:
            break

