import torch
import torch.nn as nn

class Pooling(nn.Module):
    def __init__(self, emb_dim: int, pooling_mode_mean_tokens: bool = True,
                 pooling_mode_max_tokens: bool = False,
                 pooling_cat_linear: bool = False):
        super(Pooling, self).__init__()
        self.emb_dim = emb_dim
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_cat_linear = pooling_cat_linear
        self.pooling_linear = nn.Linear(3*emb_dim,emb_dim)

        pooling_multiplier = sum([pooling_mode_max_tokens, pooling_mode_mean_tokens, pooling_cat_linear])
        self.pooling_output_dimension = (pooling_multiplier * emb_dim)
    def forward(self,features):
        '''
        :param features:[batch_size,3,dim]
        :return:[batch_size,dim]
        '''
        output_vectors = []
        if self.pooling_mode_max_tokens:
            max_over_time = torch.max(features,1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens:
            sum_emb = torch.sum(features,1)
            output_vectors.append(sum_emb / 3)
        if self.pooling_cat_linear:
            cat_emb = torch.cat(features,1)
            linear_emb = self.pooling_linear(cat_emb)
            output_vectors.append(linear_emb)

        output_vector = torch.cat(output_vectors, 1)
        return output_vector

