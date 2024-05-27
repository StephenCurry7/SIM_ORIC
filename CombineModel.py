import os
import gc
from sklearn.metrics import log_loss, mean_squared_error
import torch
from torch import nn #from tensorflow import keras
from torch.nn import Module #from tensorflow.python.keras import Model
from deepctr_torch.models import xDeepFM, DeepFM, WDL, DCN, AutoInt #from deepctr.models import xDeepFM, DeepFM, WDL, DCN, AutoInt
from deepctr_torch.models.basemodel import BaseModel
import copy #from tensorflow.python.keras.models import clone_model
from deepctr_torch.inputs import combined_dnn_input
import numpy as np
import copy

class CombinedModel(BaseModel):
        def __init__(self, base_model, inter_model, task, feature_names_num,linear_feature_columns, dnn_feature_columns,device='cuda:0'):
            super(CombinedModel, self).__init__(linear_feature_columns, dnn_feature_columns,device='cuda:0')

            base_part = copy.deepcopy(base_model)
            base_part.load_state_dict(base_model.state_dict())
            for param in base_part.parameters():
                param.requires_grad = False #冻结base_model，不训练这一部分（同论文，先训练interaction部分，之后再fine_tune整个集成模型）

            inter_part = copy.deepcopy(inter_model)
            inter_part.load_state_dict(inter_model.state_dict())
            
            self.base_part = base_part
            self.inter_part = inter_part
            self.feature_names_num = feature_names_num#原特征个数
            self.task = task
            
        def forward(self, input): 
            base_input = input[:,:self.feature_names_num] #base模型的输入
            inter_input = input[:,self.feature_names_num:] #inter模型的输入
            
            #求base_input的倒数第二层输出(根据deepfm的代码写)
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
                logit += dnn_logit #此处得到的logit为deepfm倒数第二层的输出
            
            base_output = logit.squeeze()
            
            #求base_input的倒数第二层输出
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
                logit += dnn_logit #此处得到的logit为deepfm倒数第二层的输出
            
            inter_output = logit.squeeze()

            x = torch.add(base_output, inter_output) #拼接两个模型的输出
            if self.task == "binary":
                x = torch.sigmoid(x)
            final_output = x.view(-1, 1) #模型的预测概率

            
            return final_output