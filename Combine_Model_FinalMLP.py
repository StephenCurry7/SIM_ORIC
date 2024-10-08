import os
import gc
from sklearn.metrics import log_loss, mean_squared_error
import torch
from torch import nn 
from torch.nn import Module
from deepctr_torch.models.basemodel import BaseModel
import copy 
from deepctr_torch.inputs import combined_dnn_input
import numpy as np
import copy

class Combined_DualMLP_Model(BaseModel):
        def __init__(self, base_model, inter_model, task, feature_names_num,linear_feature_columns, dnn_feature_columns,output_use_fusion,device='cuda:0'):
            super(Combined_DualMLP_Model, self).__init__(linear_feature_columns, dnn_feature_columns,device='cuda:0')

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
            
            self.output_use_fusion = output_use_fusion
            if self.output_use_fusion == True:
                self.output_fusion_module = InteractionAggregation(1, 1, output_dim=1, num_heads=1)
            
        def forward(self, input): 
            base_input = input[:,:self.feature_names_num] 
            inter_input = input[:,self.feature_names_num:] 

            #base_part
            base_sparse_embedding_list, base_dense_value_list = self.base_part.input_from_feature_columns(base_input, self.base_part.dnn_feature_columns,
                                                                                  self.base_part.embedding_dict) 
        
            base_mlp_input = combined_dnn_input(
                    base_sparse_embedding_list, base_dense_value_list) 

            base_logit = self.base_part.mlp1(base_mlp_input)
            base_logit += self.base_part.mlp2(base_mlp_input)
                
            #inter_part
            inter_sparse_embedding_list, inter_dense_value_list = self.inter_part.input_from_feature_columns(inter_input, self.inter_part.dnn_feature_columns,
                                                                                  self.inter_part.embedding_dict) 
        
            inter_mlp_input = combined_dnn_input(
                    inter_sparse_embedding_list, inter_dense_value_list) 

            inter_logit = self.inter_part.mlp1(inter_mlp_input)
            inter_logit += self.inter_part.mlp2(inter_mlp_input)
            
            if self.output_use_fusion == True:
                y_pred = self.output_fusion_module(base_logit, inter_logit) 
            else:
                y_pred = torch.add(base_logit, inter_logit)
            if self.task == "binary":
                y_pred = torch.sigmoid(y_pred)
            final_output = y_pred.view(-1, 1) 

            
            return final_output

class Combined_FinalMLP_Model(BaseModel):
        def __init__(self, base_model, inter_model, task, feature_names_num,linear_feature_columns, dnn_feature_columns,output_use_fusion,device='cuda:0'):
            super(Combined_FinalMLP_Model, self).__init__(linear_feature_columns, dnn_feature_columns,device='cuda:0')

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
            
            self.output_use_fusion = output_use_fusion
            if self.output_use_fusion == True:
                self.output_fusion_module = InteractionAggregation(1, 1, output_dim=1, num_heads=1)
            
        def forward(self, input): 
            base_input = input[:,:self.feature_names_num] 
            inter_input = input[:,self.feature_names_num:] 

            #base_part
            base_sparse_embedding_list, base_dense_value_list = self.base_part.input_from_feature_columns(base_input, self.base_part.dnn_feature_columns,
                                                                                  self.base_part.embedding_dict) 
    
            base_embedding_output = combined_dnn_input(base_sparse_embedding_list, base_dense_value_list)
            
            if self.base_part.use_fs:
                if len(self.base_part.fs1_context)>0:
                    base_select_sparse_embedding_list_1, base_select_dense_value_list_1 = self.base_part.input_from_feature_columns(base_input, self.base_part.fs1_dnn_feature_columns,self.base_part.embedding_dict) 
                    base_fs_input_1 = combined_dnn_input(base_select_sparse_embedding_list_1, base_select_dense_value_list_1).flatten(start_dim=1).cuda()
                else:
                    base_fs1_ctx_bias = nn.Parameter(torch.zeros(1, self.base_part.embedding_dim))
                    base_fs_input_1 = base_fs1_ctx_bias.repeat(base_embedding_output.size(0), 1).cuda()
                    
                if len(self.base_part.fs2_context)>0:
                    base_select_sparse_embedding_list_2, base_select_dense_value_list_2 = self.base_part.input_from_feature_columns(base_input, self.base_part.fs2_dnn_feature_columns,self.base_part.embedding_dict) 
                    base_fs_input_2 = combined_dnn_input(base_select_sparse_embedding_list_2, base_select_dense_value_list_2).flatten(start_dim=1).cuda()
                else:
                    base_fs2_ctx_bias = nn.Parameter(torch.zeros(1, self.base_part.embedding_dim))
                    base_fs_input_2 = base_fs2_ctx_bias.repeat(base_embedding_output.size(0), 1).cuda()
                    
    
                base_feat1, base_feat2 = self.base_part.fs_module(base_fs_input_1, base_fs_input_2, base_embedding_output)
            else:
                base_feat1, base_feat2 = base_embedding_output, base_embedding_output
                
            base_logit = self.base_part.fusion_module(self.base_part.mlp1(base_feat1), self.base_part.mlp2(base_feat2)) 

            #inter_part
            inter_sparse_embedding_list, inter_dense_value_list = self.inter_part.input_from_feature_columns(inter_input, self.inter_part.dnn_feature_columns,
                                                                                  self.inter_part.embedding_dict) 
    
            inter_embedding_output = combined_dnn_input(inter_sparse_embedding_list, inter_dense_value_list) 
            
            if self.inter_part.use_fs:
                if len(self.inter_part.fs1_context)>0:
                    inter_select_sparse_embedding_list_1, inter_select_dense_value_list_1 = self.inter_part.input_from_feature_columns(inter_input, self.inter_part.fs1_dnn_feature_columns,self.inter_part.embedding_dict) 
                    inter_fs_input_1 = combined_dnn_input(inter_select_sparse_embedding_list_1, inter_select_dense_value_list_1).flatten(start_dim=1).cuda()
                else:
                    inter_fs1_ctx_bias = nn.Parameter(torch.zeros(1, self.inter_part.embedding_dim))
                    inter_fs_input_1 = inter_fs1_ctx_bias.repeat(inter_embedding_output.size(0), 1).cuda()
                    
                if len(self.inter_part.fs2_context)>0:
                    inter_select_sparse_embedding_list_2, inter_select_dense_value_list_2 = self.inter_part.input_from_feature_columns(inter_input, self.inter_part.fs2_dnn_feature_columns,self.inter_part.embedding_dict) 
                    inter_fs_input_2 = combined_dnn_input(inter_select_sparse_embedding_list_2, inter_select_dense_value_list_2).flatten(start_dim=1).cuda()
                else:
                    inter_fs2_ctx_bias = nn.Parameter(torch.zeros(1, self.inter_part.embedding_dim))
                    inter_fs_input_2 = inter_fs2_ctx_bias.repeat(inter_embedding_output.size(0), 1).cuda()
                    
    
                inter_feat1, inter_feat2 = self.inter_part.fs_module(inter_fs_input_1, inter_fs_input_2, inter_embedding_output)
            else:
                inter_feat1, inter_feat2 = inter_embedding_output, inter_embedding_output
                
            inter_logit = self.inter_part.fusion_module(self.inter_part.mlp1(inter_feat1), self.inter_part.mlp2(inter_feat2)) 
                
    
            if self.output_use_fusion == True:
                y_pred = self.output_fusion_module(base_logit, inter_logit) 
            else:
                y_pred = torch.add(base_logit, inter_logit) 
            if self.task == "binary":
                y_pred = torch.sigmoid(y_pred)
            final_output = y_pred.view(-1, 1) 

            
            return final_output

class InteractionAggregation(nn.Module):
    def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
        super(InteractionAggregation, self).__init__()
        assert x_dim % num_heads == 0 and y_dim % num_heads == 0, \
            "Input dim must be divisible by num_heads!"
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_x_dim = x_dim // num_heads
        self.head_y_dim = y_dim // num_heads
        self.w_x = nn.Linear(x_dim, output_dim)
        self.w_y = nn.Linear(y_dim, output_dim)
        self.w_xy = nn.Parameter(torch.Tensor(num_heads * self.head_x_dim * self.head_y_dim, 
                                              output_dim))
        nn.init.xavier_normal_(self.w_xy)

    def forward(self, x, y):
        output = self.w_x(x) + self.w_y(y)
        head_x = x.view(-1, self.num_heads, self.head_x_dim)
        head_y = y.view(-1, self.num_heads, self.head_y_dim)
        xy = torch.matmul(torch.matmul(head_x.unsqueeze(2), 
                                       self.w_xy.view(self.num_heads, self.head_x_dim, -1)) \
                               .view(-1, self.num_heads, self.output_dim, self.head_y_dim),
                          head_y.unsqueeze(-1)).squeeze(-1)
        output += xy.sum(dim=1)
        return output
