import os
import gc
import torch
from torch import nn #from tensorflow import keras
from torch.nn import Module #from tensorflow.python.keras import Model
from 模块fuxictr.pytorch.models.base_model import BaseModel
from 模块fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block

import copy
import numpy as np
import logging

class Combined_DualMLP_Model(BaseModel):#仿照BaseModel写各个函数
        def __init__(self, 
                     base_model, 
                     inter_model, 
                     feature_map,
                     output_use_fusion,
                     learning_rate=1e-3,
                     evaluate=True,#用于判断是否需要检验模型
                     **kwargs):
            super(Combined_DualMLP_Model, self).__init__(feature_map,**kwargs)
            

            base_part = copy.deepcopy(base_model)
            base_part.load_state_dict(base_model.state_dict())
            for param in base_part.parameters():
                param.requires_grad = False #冻结base_model，不训练这一部分（同论文，先训练interaction部分，之后再fine_tune整个集成模型）

            inter_part = copy.deepcopy(inter_model)
            inter_part.load_state_dict(inter_model.state_dict())
            
            self.base_part = base_part
            self.inter_part = inter_part

            self.output_use_fusion = output_use_fusion
            if self.output_use_fusion == True:
                self.output_fusion_module = InteractionAggregation(1, 1, output_dim=1, num_heads=1)

            self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
            self.model_to_device()

        def combine_fit(self, data_generator, epochs=1, validation_data=None, unfreeze_base_model=False, model_path=None, weight_path=None,
            max_gradient_norm=10., **kwargs): 
            self.unfreeze_base_model = unfreeze_base_model
            #self.batch_size = batch_size
            self.valid_data = validation_data
            self.batch_size = kwargs['batch_size']
            self._max_gradient_norm = max_gradient_norm
            self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
            self._stopping_steps = 0
            self._steps_per_epoch = len(data_generator[0])
            self._stop_training = False
            self._total_steps = 0
            self._batch_index = 0
            self._epoch_index = 0
            self._model_path = model_path
            self._weight_path = weight_path
            
            
            if self._model_path == None or self._weight_path == None:
                print('No model path or weight path given!')
                print(aaaaaaaaaa)

            if self._eval_steps is None:
                self._eval_steps = self._steps_per_epoch

            logging.info("Start combine training: {} batches/epoch".format(self._steps_per_epoch))
            logging.info("************ Epoch=1 start ************")
            for epoch in range(epochs): #epochs=100 for DualMLP
                self._epoch_index = epoch
                self.combine_train_epoch(data_generator, unfreeze_base_model)
                if self._stop_training:
                    break
                else:
                    logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
            logging.info("Training finished.")
            logging.info("Load best model: {}".format(self._weight_path))#logging.info("Load best model: {}".format(self.checkpoint))
            self.load_weights(self._weight_path) #self.load_weights(self.checkpoint) #加载训练中最好的一轮模型对应的权重

        def combine_train_epoch(self, data_generator, unfreeze_base_model):
            self._batch_index = 0
            train_loss = 0
            if unfreeze_base_model == True:
                self.base_part.train()
            self.inter_part.train()
            train_gen, inter_train_gen = data_generator[0], data_generator[1]
            num = len(train_gen)#也等于len(inter_train_gen)
            '''if num % 2 == 0:
                num = int(num/2)
            else:
                num = int(num/2)+1'''
            for idx in range(num):
                self._batch_index = idx
                self._total_steps += 1
                train_batch_data = train_gen.dataset.darray[idx*self.batch_size:(idx+1)*self.batch_size]
                inter_train_batch_data = inter_train_gen.dataset.darray[idx*self.batch_size:(idx+1)*self.batch_size]
                if len(train_batch_data) > len(inter_train_batch_data):
                    train_batch_data = train_batch_data[:len(inter_train_batch_data),:]
                if len(train_batch_data) < len(inter_train_batch_data):
                    inter_train_batch_data = inter_train_batch_data[:len(train_batch_data),:]
                batch_data = [train_batch_data, inter_train_batch_data]
                loss = self.combine_train_step(batch_data)
                train_loss += loss.item()
                if self._total_steps % self._eval_steps == 0:
                    logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                    train_loss = 0
                    if self._evaluate == True: #我自己加的，有时候不需要验证模型，不需要下面这一步
                        self.combine_eval_step(unfreeze_base_model)
                if self._stop_training:
                    break
                    
        def combine_train_step(self, batch_data):
            self.optimizer.zero_grad()
            loss = self.get_total_combine_loss(batch_data)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            return loss

        def get_total_combine_loss(self, inputs):
            total_loss = self.add_combine_loss(inputs) + self.add_regularization()
            return total_loss

        def add_combine_loss(self, inputs):
            return_dict = self.forward(inputs)#forward需要改
            y_true = self.get_labels(inputs[0]) #inputs[0] = train_batch_data,inputs[1] = inter_train_batch_data，二者的label一样
            loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
            return loss

        def combine_eval_step(self, unfreeze_base_model):
            logging.info('Evaluation @epoch {} - batch {}: '.format(self._epoch_index + 1, self._batch_index + 1))
            val_logs = self.combine_evaluate(self.valid_data, metrics=self._monitor.get_metrics())
            self.checkpoint_and_earlystop(val_logs)
            if unfreeze_base_model == True:
                self.base_part.train()
            self.inter_part.train()

        def combine_evaluate(self, data_generator, metrics=None):
            self.eval()  # set to evaluation mode
            if len(data_generator) == 4:
                with torch.no_grad():
                    y_pred = []
                    y_true = []
                    group_id = []
                    valid_gen_1, inter_valid_gen_1 = data_generator[0], data_generator[1]
                    valid_gen_2, inter_valid_gen_2 = data_generator[2], data_generator[3]
                    num_1 = len(valid_gen_1)#也等于len(inter_valid_gen_1)
                    num_2 = len(valid_gen_2)#也等于len(inter_valid_gen_2)
                    '''if num_1 % 2 == 0:
                        num_1 = int(num_1/2)
                    else:
                        num_1 = int(num_1/2)+1

                    if num_2 % 2 == 0:
                        num_2 = int(num_2/2)
                    else:
                        num_2 = int(num_2/2)+1'''
        
                    for idx in range(num_1):
                        valid_batch_data = valid_gen_1.dataset.darray[idx*self.batch_size:(idx+1)*self.batch_size]
                        inter_valid_batch_data = inter_valid_gen_1.dataset.darray[idx*self.batch_size:(idx+1)*self.batch_size]
                        if len(valid_batch_data) > len(inter_valid_batch_data):
                            valid_batch_data = valid_batch_data[:len(inter_valid_batch_data),:]
                        if len(valid_batch_data) < len(inter_valid_batch_data):
                            inter_valid_batch_data = inter_valid_batch_data[:len(valid_batch_data),:]
                        batch_data = [valid_batch_data, inter_valid_batch_data]
                        return_dict = self.forward(batch_data)
                        y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                        y_true.extend(self.get_labels(batch_data[0]).data.cpu().numpy().reshape(-1))
                        if self.feature_map.group_id is not None:
                            print('feature_map.group_id is not None!!')
                            print('aaaa')
                            group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
                    for idx in range(num_2):
                        valid_batch_data = valid_gen_2.dataset.darray[idx*self.batch_size:(idx+1)*self.batch_size]
                        inter_valid_batch_data = inter_valid_gen_2.dataset.darray[idx*self.batch_size:(idx+1)*self.batch_size]
                        if len(valid_batch_data) > len(inter_valid_batch_data):
                            valid_batch_data = valid_batch_data[:len(inter_valid_batch_data),:]
                        if len(valid_batch_data) < len(inter_valid_batch_data):
                            inter_valid_batch_data = inter_valid_batch_data[:len(valid_batch_data),:]
                        batch_data = [valid_batch_data, inter_valid_batch_data]
                        return_dict = self.forward(batch_data)
                        y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                        y_true.extend(self.get_labels(batch_data[0]).data.cpu().numpy().reshape(-1))
                        if self.feature_map.group_id is not None:
                            print('feature_map.group_id is not None!!')
                            print('aaaa')
                            group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
                        
                    y_pred = np.array(y_pred, np.float64)
                    y_true = np.array(y_true, np.float64)
                    group_id = np.array(group_id) if len(group_id) > 0 else None
                    if metrics is not None:
                        val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
                    else:
                        val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)
                    logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
                    return val_logs
            else:
                with torch.no_grad():
                    y_pred = []
                    y_true = []
                    group_id = []
                    valid_gen, inter_valid_gen = data_generator[0], data_generator[1]
                    num = len(valid_gen)#也等于len(inter_valid_gen)
                    '''if num % 2 == 0:
                        num = int(num/2)
                    else:
                        num = int(num/2)+1'''
                    for idx in range(num):
                        valid_batch_data = valid_gen.dataset.darray[idx*self.batch_size:(idx+1)*self.batch_size]
                        inter_valid_batch_data = inter_valid_gen.dataset.darray[idx*self.batch_size:(idx+1)*self.batch_size]
                        if len(valid_batch_data) > len(inter_valid_batch_data):
                            valid_batch_data = valid_batch_data[:len(inter_valid_batch_data),:]
                        if len(valid_batch_data) < len(inter_valid_batch_data):
                            inter_valid_batch_data = inter_valid_batch_data[:len(valid_batch_data),:]
                        batch_data = [valid_batch_data, inter_valid_batch_data]
                        return_dict = self.forward(batch_data)
                        y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                        y_true.extend(self.get_labels(batch_data[0]).data.cpu().numpy().reshape(-1))
                        if self.feature_map.group_id is not None:
                            print('feature_map.group_id is not None!!')
                            print('aaaa')
                            group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
                    y_pred = np.array(y_pred, np.float64)
                    y_true = np.array(y_true, np.float64)
                    group_id = np.array(group_id) if len(group_id) > 0 else None
                    if metrics is not None:
                        val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
                    else:
                        val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)
                    logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
                    return val_logs
    
        def forward(self, inputs): #这里的forward是根据DualMLP的代码写的，不同模型不一样
            base_input, inter_input = inputs[0], inputs[1]

            base_X = self.base_part.get_inputs(base_input)
            base_flat_emb = self.base_part.embedding_layer(base_X).flatten(start_dim=1)
            base_y_pred = self.base_part.mlp1(base_flat_emb) + self.base_part.mlp2(base_flat_emb)

            inter_X = self.inter_part.get_inputs(inter_input)
            inter_flat_emb = self.inter_part.embedding_layer(inter_X).flatten(start_dim=1)
            inter_y_pred = self.inter_part.mlp1(inter_flat_emb) + self.inter_part.mlp2(inter_flat_emb)

            if self.output_use_fusion == True:
                y_pred = self.output_fusion_module(base_y_pred, inter_y_pred) #二维拼接
            else:
                y_pred = torch.add(base_y_pred, inter_y_pred)#拼接两个模型的输出
            y_pred = self.output_activation(y_pred) #模型的预测概率
            return_dict = {"y_pred": y_pred}
            return return_dict


class Combined_FinalMLP_Model(BaseModel):#仿照BaseModel写各个函数
        def __init__(self, 
                     base_model, 
                     inter_model, 
                     feature_map,
                     output_use_fusion,
                     learning_rate=1e-3,
                     evaluate=True,#用于判断是否需要检验模型
                     **kwargs):
            super(Combined_FinalMLP_Model, self).__init__(feature_map,**kwargs)
            

            base_part = copy.deepcopy(base_model)
            base_part.load_state_dict(base_model.state_dict())
            for param in base_part.parameters():
                param.requires_grad = False #冻结base_model，不训练这一部分（同论文，先训练interaction部分，之后再fine_tune整个集成模型）

            inter_part = copy.deepcopy(inter_model)
            inter_part.load_state_dict(inter_model.state_dict())
            
            self.base_part = base_part
            self.inter_part = inter_part

            self.output_use_fusion = output_use_fusion
            if self.output_use_fusion == True:
                self.output_fusion_module = InteractionAggregation(1, 1, output_dim=1, num_heads=1)

            self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
            self.model_to_device()

        def combine_fit(self, data_generator, epochs=1, validation_data=None, unfreeze_base_model=False, model_path=None, weight_path=None,
            max_gradient_norm=10., **kwargs): 
            self.unfreeze_base_model = unfreeze_base_model
            #self.batch_size = batch_size
            self.valid_data = validation_data
            self.batch_size = kwargs['batch_size']
            self._max_gradient_norm = max_gradient_norm
            self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
            self._stopping_steps = 0
            self._steps_per_epoch = len(data_generator[0])
            self._stop_training = False
            self._total_steps = 0
            self._batch_index = 0
            self._epoch_index = 0
            self._model_path = model_path
            self._weight_path = weight_path
            
            
            if self._model_path == None or self._weight_path == None:
                print('No model path or weight path given!')
                print(aaaaaaaaaa)

            if self._eval_steps is None:
                self._eval_steps = self._steps_per_epoch

            logging.info("Start combine training: {} batches/epoch".format(self._steps_per_epoch))
            logging.info("************ Epoch=1 start ************")
            for epoch in range(epochs): #epochs=100 for DualMLP
                self._epoch_index = epoch
                self.combine_train_epoch(data_generator, unfreeze_base_model)
                if self._stop_training:
                    break
                else:
                    logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
            logging.info("Training finished.")
            logging.info("Load best model: {}".format(self._weight_path))#logging.info("Load best model: {}".format(self.checkpoint))
            self.load_weights(self._weight_path) #self.load_weights(self.checkpoint) #加载训练中最好的一轮模型对应的权重

        def combine_train_epoch(self, data_generator, unfreeze_base_model):
            self._batch_index = 0
            train_loss = 0
            if unfreeze_base_model == True:
                self.base_part.train()
            self.inter_part.train()
            train_gen, inter_train_gen = data_generator[0], data_generator[1]
            num = len(train_gen)#也等于len(inter_train_gen)
            '''if num % 2 == 0:
                num = int(num/2)
            else:
                num = int(num/2)+1'''
            for idx in range(num):
                self._batch_index = idx
                self._total_steps += 1
                train_batch_data = train_gen.dataset.darray[idx*self.batch_size:(idx+1)*self.batch_size]
                inter_train_batch_data = inter_train_gen.dataset.darray[idx*self.batch_size:(idx+1)*self.batch_size]
                if len(train_batch_data) > len(inter_train_batch_data):
                    train_batch_data = train_batch_data[:len(inter_train_batch_data),:]
                if len(train_batch_data) < len(inter_train_batch_data):
                    inter_train_batch_data = inter_train_batch_data[:len(train_batch_data),:]
                batch_data = [train_batch_data, inter_train_batch_data]
                loss = self.combine_train_step(batch_data)
                train_loss += loss.item()
                if self._total_steps % self._eval_steps == 0:
                    logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                    train_loss = 0
                    if self._evaluate == True: #我自己加的，有时候不需要验证模型，不需要下面这一步
                        self.combine_eval_step(unfreeze_base_model)
                if self._stop_training:
                    break
                    
        def combine_train_step(self, batch_data):
            self.optimizer.zero_grad()
            loss = self.get_total_combine_loss(batch_data)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            return loss

        def get_total_combine_loss(self, inputs):
            total_loss = self.add_combine_loss(inputs) + self.add_regularization()
            return total_loss

        def add_combine_loss(self, inputs):
            return_dict = self.forward(inputs)
            y_true = self.get_labels(inputs[0]) #inputs[0] = train_batch_data,inputs[1] = inter_train_batch_data，二者的label一样
            loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
            return loss

        def combine_eval_step(self, unfreeze_base_model):
            logging.info('Evaluation @epoch {} - batch {}: '.format(self._epoch_index + 1, self._batch_index + 1))
            val_logs = self.combine_evaluate(self.valid_data, metrics=self._monitor.get_metrics())
            self.checkpoint_and_earlystop(val_logs)
            if unfreeze_base_model == True:
                self.base_part.train()
            self.inter_part.train()

        def combine_evaluate(self, data_generator, metrics=None):
            self.eval()  # set to evaluation mode
            if len(data_generator) == 4:
                with torch.no_grad():
                    y_pred = []
                    y_true = []
                    group_id = []
                    valid_gen_1, inter_valid_gen_1 = data_generator[0], data_generator[1]
                    valid_gen_2, inter_valid_gen_2 = data_generator[2], data_generator[3]
                    num_1 = len(valid_gen_1)#也等于len(inter_valid_gen_1)
                    num_2 = len(valid_gen_2)#也等于len(inter_valid_gen_2)
                    '''if num_1 % 2 == 0:
                        num_1 = int(num_1/2)
                    else:
                        num_1 = int(num_1/2)+1

                    if num_2 % 2 == 0:
                        num_2 = int(num_2/2)
                    else:
                        num_2 = int(num_2/2)+1'''
                    for idx in range(num_1):
                        valid_batch_data = valid_gen_1.dataset.darray[idx*self.batch_size:(idx+1)*self.batch_size]
                        inter_valid_batch_data = inter_valid_gen_1.dataset.darray[idx*self.batch_size:(idx+1)*self.batch_size]
                        if len(valid_batch_data) > len(inter_valid_batch_data):
                            valid_batch_data = valid_batch_data[:len(inter_valid_batch_data),:]
                        if len(valid_batch_data) < len(inter_valid_batch_data):
                            inter_valid_batch_data = inter_valid_batch_data[:len(valid_batch_data),:]
                        batch_data = [valid_batch_data, inter_valid_batch_data]
                        return_dict = self.forward(batch_data)
                        y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                        y_true.extend(self.get_labels(batch_data[0]).data.cpu().numpy().reshape(-1))
                        if self.feature_map.group_id is not None:
                            print('feature_map.group_id is not None!!')
                            print('aaaa')
                            group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
                    for idx in range(num_2):
                        valid_batch_data = valid_gen_2.dataset.darray[idx*self.batch_size:(idx+1)*self.batch_size]
                        inter_valid_batch_data = inter_valid_gen_2.dataset.darray[idx*self.batch_size:(idx+1)*self.batch_size]
                        if len(valid_batch_data) > len(inter_valid_batch_data):
                            valid_batch_data = valid_batch_data[:len(inter_valid_batch_data),:]
                        if len(valid_batch_data) < len(inter_valid_batch_data):
                            inter_valid_batch_data = inter_valid_batch_data[:len(valid_batch_data),:]
                        batch_data = [valid_batch_data, inter_valid_batch_data]
                        return_dict = self.forward(batch_data)
                        y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                        y_true.extend(self.get_labels(batch_data[0]).data.cpu().numpy().reshape(-1))
                        
                    y_pred = np.array(y_pred, np.float64)
                    y_true = np.array(y_true, np.float64)
                    group_id = np.array(group_id) if len(group_id) > 0 else None
                    if metrics is not None:
                        val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
                    else:
                        val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)
                    logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
                    return val_logs
            else:
                with torch.no_grad():
                    y_pred = []
                    y_true = []
                    group_id = []
                    valid_gen, inter_valid_gen = data_generator[0], data_generator[1]
                    num = len(valid_gen)#也等于len(inter_valid_gen)
                    '''if num % 2 == 0:
                        num = int(num/2)
                    else:
                        num = int(num/2)+1'''
                    for idx in range(num):
                        valid_batch_data = valid_gen.dataset.darray[idx*self.batch_size:(idx+1)*self.batch_size]
                        inter_valid_batch_data = inter_valid_gen.dataset.darray[idx*self.batch_size:(idx+1)*self.batch_size]
                        if len(valid_batch_data) > len(inter_valid_batch_data):
                            valid_batch_data = valid_batch_data[:len(inter_valid_batch_data),:]
                        if len(valid_batch_data) < len(inter_valid_batch_data):
                            inter_valid_batch_data = inter_valid_batch_data[:len(valid_batch_data),:]
                        batch_data = [valid_batch_data, inter_valid_batch_data]
                        return_dict = self.forward(batch_data)
                        y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                        y_true.extend(self.get_labels(batch_data[0]).data.cpu().numpy().reshape(-1))
                        if self.feature_map.group_id is not None:
                            print('feature_map.group_id is not None!!')
                            print('aaaa')
                            group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
                    y_pred = np.array(y_pred, np.float64)
                    y_true = np.array(y_true, np.float64)
                    group_id = np.array(group_id) if len(group_id) > 0 else None
                    if metrics is not None:
                        val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
                    else:
                        val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)
                    logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
                    return val_logs
    
        def forward(self, inputs): #这里的forward是根据FianlMLP的代码写的，不同模型不一样
            base_input, inter_input = inputs[0], inputs[1]

            base_X = self.base_part.get_inputs(base_input)
            base_flat_emb = self.base_part.embedding_layer(base_X).flatten(start_dim=1)
            if self.base_part.use_fs:
                base_feat1, base_feat2 = self.base_part.fs_module(base_X, base_flat_emb)
            else:
                base_feat1, base_feat2 = base_flat_emb, base_flat_emb
            base_y_pred = self.base_part.fusion_module(self.base_part.mlp1(base_feat1), self.base_part.mlp2(base_feat2))

            inter_X = self.inter_part.get_inputs(inter_input)
            inter_flat_emb = self.inter_part.embedding_layer(inter_X).flatten(start_dim=1)
            if self.inter_part.use_fs:
                inter_feat1, inter_feat2 = self.inter_part.fs_module(inter_X, inter_flat_emb)
            else:
                inter_feat1, inter_feat2 = inter_flat_emb, inter_flat_emb
            inter_y_pred = self.inter_part.fusion_module(self.inter_part.mlp1(inter_feat1), self.inter_part.mlp2(inter_feat2))

            if self.output_use_fusion == True:
                y_pred = self.output_fusion_module(base_y_pred, inter_y_pred) #二维拼接
            else:
                y_pred = torch.add(base_y_pred, inter_y_pred)#拼接两个模型的输出
            y_pred = self.output_activation(y_pred)#模型的预测概率
            return_dict = {"y_pred": y_pred}
            return return_dict
            

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