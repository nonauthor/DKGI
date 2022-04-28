
import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch.conv import SAGEConv
from layers.discriminator import Discriminator
from layers.readout import AvgReadout
from layers.kbgat_encoder import SpKBGATModified

class DKGI(nn.Module):
    def __init__(self, out_dim, entity_embeddings, relation_embeddings):
        super(DKGI,self).__init__()
        self.kbgat = SpKBGATModified(entity_embeddings, relation_embeddings, entity_out_dim=[150,300],
                                     relation_out_dim=[150,300],
                                     drop_GAT=0.3, alpha=0.2, nheads_GAT=[2,2])

        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(out_dim)


    def forward(self, Corpus_, msk, samp_bias1, samp_bias2, train_indices,current_batch_2hop_indices, rel_DIM):
        h_1,r_1 = self.kbgat(Corpus_, Corpus_.train_adj_matrix, train_indices, current_batch_2hop_indices, shuffle=False)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2,r_2 = self.kbgat(Corpus_, Corpus_.train_adj_matrix, train_indices, current_batch_2hop_indices, shuffle='entity,relation feature')

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        if rel_DIM:
            c_rel = self.read(r_1,msk)
            ret_rel = self.disc(c_rel, r_1, r_2, samp_bias1, samp_bias2)
        else: ret_rel = None

        return ret, ret_rel

    # Detach the return variables
    def embed(self, Corpus_, msk, train_indices, current_batch_2hop_indices):
        h_1,r_1 = self.kbgat(Corpus_, Corpus_.train_adj_matrix, train_indices, current_batch_2hop_indices, shuffle=False)
        c = self.read(h_1, msk)

        return h_1.detach(), r_1.detach(), c.detach()