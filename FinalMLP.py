import torch
from torch import nn
from fuxictr.pytorch.layers.blocks.mlp_block import MLP_Block
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import DNN

class FinalMLP(BaseModel):
    def __init__(self,
                 sparse_feat,
                 dense_feat,
                 linear_feature_columns, dnn_feature_columns, 
                 task='binary', device='cuda:0',use_fm=True,
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024, gpus=None,
                 #learning_rate=1e-3,
                 embedding_dim=10,
                 mlp1_hidden_units=[64, 64, 64],
                 mlp1_hidden_activations="ReLU",
                 mlp1_dropout=0,
                 mlp1_batch_norm=False,
                 mlp2_hidden_units=[64, 64, 64],
                 mlp2_hidden_activations="ReLU",
                 mlp2_dropout=0,
                 mlp2_batch_norm=False,
                 use_fs=True,
                 fs_hidden_units=[64],
                 fs1_context=[],
                 fs2_context=[],
                 num_heads=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 batch_size=4092):
        super(FinalMLP, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)
        feature_dim = embedding_dim * len(sparse_feat) + len(dense_feat)
        self.embedding_dim = embedding_dim
        self.fs1_context = fs1_context
        self.fs2_context = fs2_context

        self.mlp1 = MLP_Block(input_dim=self.compute_input_dim(dnn_feature_columns),
                              output_dim=None, 
                              hidden_units=mlp1_hidden_units,
                              hidden_activations=mlp1_hidden_activations,
                              output_activation=None,
                              dropout_rates=mlp1_dropout,
                              batch_norm=mlp1_batch_norm)
        self.mlp2 = MLP_Block(input_dim=self.compute_input_dim(dnn_feature_columns),
                              output_dim=None, 
                              hidden_units=mlp2_hidden_units,
                              hidden_activations=mlp2_hidden_activations,
                              output_activation=None,
                              dropout_rates=mlp2_dropout, 
                              batch_norm=mlp2_batch_norm)   
        
        self.use_fs = use_fs
        if self.use_fs:
            if len(fs1_context) == 0:
                self.fs1_dnn_feature_columns = dnn_feature_columns
                fs1_gate_input_dim = embedding_dim
            else:
                self.fs1_dnn_feature_columns = []
                for each in dnn_feature_columns:
                    if each.name in fs1_context:
                        self.fs1_dnn_feature_columns.append(each)
                fs1_gate_input_dim = self.compute_input_dim(self.fs1_dnn_feature_columns)                        
            if len(fs2_context) == 0:
                self.fs2_dnn_feature_columns = dnn_feature_columns
                fs2_gate_input_dim = embedding_dim
            else:
                self.fs2_dnn_feature_columns = []
                for each in dnn_feature_columns:
                    if each.name in fs2_context:
                        self.fs2_dnn_feature_columns.append(each)
                fs2_gate_input_dim = self.compute_input_dim(self.fs2_dnn_feature_columns)          
            
                    
            self.fs_module = FeatureSelection(feature_dim, 
                                              embedding_dim, 
                                              fs1_gate_input_dim,
                                              fs2_gate_input_dim,
                                              fs_hidden_units, 
                                              fs1_context,
                                              fs2_context,
                                              device
                                              )
        self.fusion_module = InteractionAggregation(mlp1_hidden_units[-1], 
                                                    mlp2_hidden_units[-1], 
                                                    output_dim=1, 
                                                    num_heads=num_heads)
        #self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate) #compile函数外置使用
        #self.reset_parameters()
        #self.model_to_device()
            
    def forward(self, X):
        """
        Inputs: [X,y]
        """
       embedding_output = combined_dnn_input(sparse_embedding_list, dense_value_list) #shape=torch.Size([8192, 273])
        
        if self.use_fs:
            if len(self.fs1_context)>0:
                select_sparse_embedding_list_1, select_dense_value_list_1 = self.input_from_feature_columns(X, self.fs1_dnn_feature_columns,
                                                                                  self.embedding_dict) 
                fs_input_1 = combined_dnn_input(select_sparse_embedding_list_1, select_dense_value_list_1).flatten(start_dim=1).cuda()
            else:
                fs1_ctx_bias = nn.Parameter(torch.zeros(1, self.embedding_dim))
                fs_input_1 = fs1_ctx_bias.repeat(embedding_output.size(0), 1).cuda()
                
            if len(self.fs2_context)>0:
                select_sparse_embedding_list_2, select_dense_value_list_2 = self.input_from_feature_columns(X, self.fs2_dnn_feature_columns,
                                                                                      self.embedding_dict) 
                fs_input_2 = combined_dnn_input(select_sparse_embedding_list_2, select_dense_value_list_2).flatten(start_dim=1).cuda()
            else:
                fs2_ctx_bias = nn.Parameter(torch.zeros(1, self.embedding_dim))
                fs_input_2 = fs2_ctx_bias.repeat(embedding_output.size(0), 1).cuda()
                

            feat1, feat2 = self.fs_module(fs_input_1, fs_input_2, embedding_output)
        else:
            feat1, feat2 = embedding_output, embedding_output
            
        logit = self.fusion_module(self.mlp1(feat1), self.mlp2(feat2)) 
        y_pred = self.out(logit)#y_pred = self.output_activation(y_pred)
        #return_dict = {"y_pred": y_pred}
        return y_pred#return_dict


class FeatureSelection(nn.Module):
    def __init__(self, feature_dim, embedding_dim, fs1_gate_input_dim,fs2_gate_input_dim, fs_hidden_units=[], 
                 fs1_context=[], fs2_context=[],device='cuda:0'):
        super(FeatureSelection, self).__init__()
 
        self.fs1_gate = MLP_Block(input_dim=fs1_gate_input_dim,
                                  output_dim=feature_dim,
                                  hidden_units=fs_hidden_units,
                                  hidden_activations="Sigmoid",#"ReLU",
                                  output_activation=None,#"Sigmoid",
                                  batch_norm=False)
        self.fs2_gate = MLP_Block(input_dim=fs2_gate_input_dim,
                                  output_dim=feature_dim,
                                  hidden_units=fs_hidden_units,
                                  hidden_activations="Sigmoid",#"ReLU",
                                  output_activation=None,#"Sigmoid",
                                  batch_norm=False)

        
    def forward(self, fs_input_1, fs_input_2, embedding_output):
        gt1 = self.fs1_gate(fs_input_1) *2
        feature1 = embedding_output * gt1

        gt2 = self.fs2_gate(fs_input_2) *2
        feature2 = embedding_output * gt2
        return feature1, feature2


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
