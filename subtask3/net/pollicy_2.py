import torch
import torch.nn as nn
import math

from collections import OrderedDict

class PolicyNet(nn.Module):
    def __init__(self, emb_size, key_size, item_size, meta_size,
                 coordi_size, eval_node, num_rnk, use_batch_norm, 
                 use_dropout, zero_prob, use_multimodal, 
                 img_feat_size, name='PolicyNet'):
        super().__init__()
        
        # 입력 벡터 관련 변수 초기화
        self._item_size = item_size # 각 카테고리 별 개수
        self._emb_size = emb_size # 임베딩 크기
        self._key_size = key_size # 대화 요약 정보 (req 벡터 크기)
        
        # 상수 초기화
        self._name = name
        self._meta_size = meta_size
        self._coordi_size = coordi_size
        self._num_rnk = num_rnk

        # Dropout 관련 초기화
        self._use_dropout = use_dropout
        self._zero_prob = zero_prob
        
        # hidden_size 관련 초기화
        buf = eval_node[1:-1].split('][')
        self._num_hid_eval = list(map(int, buf[0].split(','))) # 코디에 대한 정보 학습
        self._num_hid_rnk = list(map(int, buf[1].split(','))) # 순위에 대한 정보 학습
        self._num_hid_layer_eval = len(self._num_hid_eval)
        self._num_hid_layer_rnk = len(self._num_hid_rnk)

        # 각 코디를 임베딩하기 위한 레이어
        self._embed = nn.ModuleList([
            nn.Embedding(size, emb_size) for size in item_size
        ])
        
        # 각 코디 임베딩을 위한 Bias
        self._bias = nn.ParameterList([
            nn.Parameter(torch.normal(mean=0.0, std=0.01, size=(1, self._embed_size)), requires_grad=True) for _ in range(self._coordi_size)
        ])

        # 각 코디에 대한 정보 학습을 위한 MLP
        num_in = self._emb_size * self._coordi_size + self._key_size
        if use_multimodal:
            num_in += img_feat_size
        
        mlp_eval_list = OrderedDict([])
        for i in range(self._num_hid_layer_eval):
            num_out = self._num_hid_eval[i]
            mlp_eval_list.update({'layer%s_linear'%(i+1): nn.Linear(num_in, num_out)})
            mlp_eval_list.update({'layer%s_relu'%(i+1): nn.ReLU()})
            if use_batch_norm:
                mlp_eval_list.update({'layer%s_bn'%(i+1): nn.BatchNorm1d(num_out)})
            if self._use_dropout:
                mlp_eval_list.update({'layer%s_dropout'%(i+1): nn.Dropout(p=self._zero_prob)})
            num_in = num_out
        
        self._eval_out_node = num_out 
        self._mlp_eval = nn.Sequential(mlp_eval_list) 

        # 순위에 대한 정보 학습을 위한 MLP
        num_in = self._eval_out_node * self._num_rnk + self._key_size
        
        mlp_rnk_list = OrderedDict([])
        for i in range(self._num_hid_layer_rnk + 1):
            if i == self._num_hid_layer_rnk:
                num_out = math.factorial(self._num_rnk)
                mlp_rnk_list.update({'layer%s_linear'%(i+1): nn.Linear(num_in, num_out)})
            else:
                num_out = self._num_hid_rnk[i]
                mlp_rnk_list.update({'layer%s_linear'%(i+1): nn.Linear(num_in, num_out)})
                mlp_rnk_list.update({'layer%s_relu'%(i+1): nn.ReLU()})
                if use_batch_norm:
                    mlp_rnk_list.update({'layer%s_bn'%(i+1): nn.BatchNorm1d(num_out)})
                if self._use_dropout:
                    mlp_rnk_list.update({'layer%s_dropout'%(i+1): nn.Dropout(p=self._zero_prob)})
            num_in = num_out
        
        self._mlp_rnk = nn.Sequential(mlp_rnk_list)
                
    def _embed_coordi(self, crd):
        embedded = []
        for i in range(self._coordi_size):
            ids = crd[:, :, i]
            embeds = self._embed[i](ids) + self._bias[i]
            embedded.append(embeds)

        embedded = torch.cat(embedded, dim=-1)
        return embedded

    def _evaluate_coordi(self, crd, req):
        crd_and_req = torch.cat((crd, req), 1)
        evl = self._mlp_eval(crd_and_req)
        return evl

    def _ranking_coordi(self, in_rnk):
        out_rnk = self._mlp_rnk(in_rnk)
        return out_rnk
    
    def forward(self, req, crd):
        crd = self._embed_coordi(crd)
        crd_tr = torch.transpose(crd, 1, 0)

        for i in range(self._num_rnk):
            crd_eval = self._evaluate_coordi(crd_tr[i], req)
            if i == 0:
                in_rnk = crd_eval
            else:
                in_rnk = torch.cat((in_rnk, crd_eval), 1)
        
        in_rnk = torch.cat((in_rnk, req), 1)
        out_rnk = self._ranking_coordi(in_rnk)
        
        return out_rnk
            
