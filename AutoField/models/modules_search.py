import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layer import *

class BasicModel(torch.nn.Module):
    def __init__(self, opt):
        super(BasicModel, self).__init__()
        self.device = torch.device("cuda:0" if opt.get('use_cuda') else "cpu")
        self.latent_dim = opt['latent_dim']
        self.field_num = len(opt['field_dim'])
        self.feature_num = sum(opt['field_dim'])
        self.field_dim = opt['field_dim']
        self.embedding = self.init_embedding()

    def init_embedding(self):
        e = nn.Parameter(torch.rand([self.feature_num, self.latent_dim]))
        nn.init.xavier_uniform_(e)
        return e

    def calc_arch_prob(self, beta, arch):
        return F.softmax(arch / beta, dim=0)

    def get_arch(self, arch):
        my_arch = arch.detach().cpu().numpy()
        current_arch = np.zeros_like(my_arch)
        current_arch = np.where(my_arch > 0, 1, 0)
        return current_arch

    def calc_sparsity(self, arch):
        base = self.latent_dim * self.feature_num
        current_arch = self.get_arch(arch)
        params = 0
        for i, num_i in enumerate(self.field_dim):
            params += num_i * current_arch[i] * self.latent_dim
        return params, (1 - params/base)

    def calc_input(self, x, beta, arch):
        xv = F.embedding(x, self.embedding)
        prob = self.calc_arch_prob(beta, arch)
        prob = prob.unsqueeze(0).unsqueeze(2)
        xe = torch.mul(xv, prob)
        return xe

class FM_search(BasicModel):
    def __init__(self, opt):
        super(FM_search, self).__init__(opt)
        self.linear = FeaturesLinear(opt['field_dim'])
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x, beta=1, arch=None):
        linear_score = self.linear.forward(x)
        xe = self.calc_input(x, beta, arch)
        fm_score = self.fm.forward(xe)
        score = linear_score + fm_score
        return score.squeeze(1)

class DeepFM_search(FM_search):
    def __init__(self, opt):
        super(DeepFM_search, self).__init__(opt)
        self.embed_output_dim = self.field_num * self.latent_dim
        self.mlp_dims = opt['mlp_dims']
        self.dropout = opt['mlp_dropout']
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, self.mlp_dims, dropout=self.dropout)

    def forward(self, x, beta=1, arch=None):
        linear_score = self.linear.forward(x)
        xe = self.calc_input(x, beta, arch)
        fm_score = self.fm.forward(xe)
        dnn_score = self.mlp.forward(xe.view(-1, self.embed_output_dim))
        score = linear_score + fm_score + dnn_score
        return score.squeeze(1)

class FNN_search(BasicModel):
    def __init__(self, opt):
        super(FNN_search, self).__init__(opt)
        self.embed_output_dim = self.field_num * self.latent_dim
        self.mlp_dims = opt['mlp_dims']
        self.dropout = opt['mlp_dropout']
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, self.mlp_dims, dropout=self.dropout)

    def forward(self, x, beta=1, arch=None):
        xv = self.calc_input(x, beta, arch)
        score = self.mlp.forward(xv.view(-1, self.embed_output_dim))
        return score.squeeze(1)

class IPNN_search(BasicModel):
    def __init__(self, opt):
        super(IPNN_search, self).__init__(opt)      
        self.embed_output_dim = self.field_num * self.latent_dim
        self.product_output_dim = int(self.field_num * (self.field_num - 1) / 2)
        self.dnn_input_dim = self.embed_output_dim + self.product_output_dim
        self.mlp_dims = opt['mlp_dims']
        self.dropout = opt['mlp_dropout']
        self.mlp = MultiLayerPerceptron(self.dnn_input_dim, self.mlp_dims, dropout=self.dropout)

        # Create indexes
        rows = []
        cols = []
        for i in range(self.field_num):
            for j in range(i+1, self.field_num):
                rows.append(i)
                cols.append(j)
        self.rows = torch.tensor(rows, device=self.device)
        self.cols = torch.tensor(cols, device=self.device)

    def calc_product(self, xe):
        batch_size = xe.shape[0]
        trans = torch.transpose(xe, 1, 2)
        gather_rows = torch.gather(trans, 2, self.rows.expand(batch_size, trans.shape[1], self.rows.shape[0]))
        gather_cols = torch.gather(trans, 2, self.cols.expand(batch_size, trans.shape[1], self.rows.shape[0]))
        p = torch.transpose(gather_rows, 1, 2)
        q = torch.transpose(gather_cols, 1, 2)
        product_embedding = torch.mul(p, q)
        product_embedding = torch.sum(product_embedding, 2)
        return product_embedding

    def forward(self, x, beta=1, arch=None):
        xv = self.calc_input(x, beta, arch)
        product = self.calc_product(xv)
        xv = xv.view(-1, self.embed_output_dim)
        xe = torch.cat((xv, product), 1)
        score = self.mlp.forward(xe)
        return score.squeeze(1)

class DCN_search(BasicModel):
    def __init__(self, opt):
        super(DCN_search, self).__init__(opt)
        self.embed_output_dim = self.field_num * self.latent_dim
        self.mlp_dims = opt['mlp_dims']
        self.dropout = opt['mlp_dropout']
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, self.mlp_dims, dropout=self.dropout, output_layer=False)
        self.cross = CrossNetwork(self.embed_output_dim, opt['cross_layer_num'])
        self.combine = torch.nn.Linear(self.mlp_dims[-1] + self.embed_output_dim, 1)

    def forward(self, x, beta=1, arch=None):
        xe = self.calc_input(x, beta, arch)
        dnn_score = self.mlp.forward(xe.view(-1, self.embed_output_dim))
        cross_score = self.cross.forward(xe.view(-1, self.embed_output_dim))
        stacked = torch.cat((dnn_score, cross_score), 1)
        score = self.combine(stacked)
        return score.squeeze(1)

