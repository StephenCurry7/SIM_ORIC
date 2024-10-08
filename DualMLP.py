import torch
from torch import nn
from fuxictr.pytorch.layers.blocks.mlp_block import MLP_Block
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import DNN

class DualMLP(BaseModel):
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, 
                 task='binary', device='cuda:0',use_fm=True,
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024, gpus=None,
                 embedding_dim=10,
                 mlp1_hidden_units=[64, 64, 64],
                 mlp1_hidden_activations="ReLU",
                 mlp1_dropout=0,
                 mlp1_batch_norm=False,
                 mlp2_hidden_units=[64, 64, 64],
                 mlp2_hidden_activations="ReLU",
                 mlp2_dropout=0,
                 mlp2_batch_norm=False,
                 fs_hidden_units=[64],
                 fs1_context=[],
                 fs2_context=[],
                 num_heads=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 batch_size=4092):
        super(DualMLP, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)
        self.mlp1 = MLP_Block(input_dim=self.compute_input_dim(dnn_feature_columns),
                              output_dim=1, 
                              hidden_units=mlp1_hidden_units,
                              hidden_activations=mlp1_hidden_activations,
                              output_activation=None,
                              dropout_rates=mlp1_dropout,
                              batch_norm=mlp1_batch_norm)
        self.mlp2 = MLP_Block(input_dim=self.compute_input_dim(dnn_feature_columns),
                              output_dim=1, 
                              hidden_units=mlp2_hidden_units,
                              hidden_activations=mlp2_hidden_activations,
                              output_activation=None,
                              dropout_rates=mlp2_dropout, 
                              batch_norm=mlp2_batch_norm)
        

        self.to(device)
            
    def forward(self, X):
        """
        Inputs: [X,y]
        """
        mlp_input = combined_dnn_input(
                sparse_embedding_list, dense_value_list) 

        logit = self.mlp1(mlp_input)
        logit += self.mlp2(mlp_input)
        
        y_pred = self.out(logit)
        return y_pred
