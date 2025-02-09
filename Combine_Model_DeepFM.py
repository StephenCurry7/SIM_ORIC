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
import copy

class CombinedModel(BaseModel):
        """Combined two DeepFM models into a SIM model"""
        def __init__(self, base_model, inter_model, task, feature_names_num,linear_feature_columns, dnn_feature_columns,device='cuda:0'):
            """Init the class"""
            super(CombinedModel, self).__init__(linear_feature_columns, dnn_feature_columns,device='cuda:0')

            base_part = copy.deepcopy(base_model)
            base_part.load_state_dict(base_model.state_dict())
            for param in base_part.parameters():
                param.requires_grad = False 

            inter_part = copy.deepcopy(inter_model)
            inter_part.load_state_dict(inter_model.state_dict())
            
            self.base_part = base_part
            self.inter_part = inter_part
            self.feature_names_num = feature_names_num
            self.task = task
            
        def forward(self, input): 
            """the forward fuction of the SIM model according to the forward 
            function of the  DeepFM model in deepctr_torch """
            base_input = input[:,:self.feature_names_num] 
            inter_input = input[:,self.feature_names_num:] 
            
            sparse_embedding_list, dense_value_list = self.base_part.input_from_feature_columns(base_input, self.base_part.dnn_feature_columns,self.base_part.embedding_dict)
            logit = self.base_part.linear_model(base_input)

            if self.base_part.use_fm and len(sparse_embedding_list) > 0:
                fm_input = torch.cat(sparse_embedding_list, dim=1)
                logit += self.base_part.fm(fm_input)

            if self.base_part.use_dnn:
                dnn_input = combined_dnn_input(
                    sparse_embedding_list, dense_value_list)
                dnn_output = self.base_part.dnn(dnn_input)
                dnn_logit = self.base_part.dnn_linear(dnn_output)
                logit += dnn_logit 
            
            base_output = logit.squeeze()
            
            sparse_embedding_list, dense_value_list = self.inter_part.input_from_feature_columns(inter_input, self.inter_part.dnn_feature_columns,self.inter_part.embedding_dict)
            logit = self.inter_part.linear_model(inter_input)

            if self.inter_part.use_fm and len(sparse_embedding_list) > 0:
                fm_input = torch.cat(sparse_embedding_list, dim=1)
                logit += self.inter_part.fm(fm_input)

            if self.inter_part.use_dnn:
                dnn_input = combined_dnn_input(
                    sparse_embedding_list, dense_value_list)
                dnn_output = self.inter_part.dnn(dnn_input)
                dnn_logit = self.inter_part.dnn_linear(dnn_output)
                logit += dnn_logit 
            
            inter_output = logit.squeeze()

            x = torch.add(base_output, inter_output) 
            if self.task == "binary":
                x = torch.sigmoid(x)
            final_output = x.view(-1, 1) 

            
            return final_output
