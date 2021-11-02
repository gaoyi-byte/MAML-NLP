import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class Embedding(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=50):
        nn.Module.__init__(self)

        self.max_length = max_length
        self.word_embedding_dim = word_embedding_dim
        
        # Word embedding
        # unk = torch.randn(1, word_embedding_dim) / math.sqrt(word_embedding_dim)
        # blk = torch.zeros(1, word_embedding_dim)
        word_vec_mat = torch.from_numpy(word_vec_mat)
        self.word_embedding = nn.Embedding(word_vec_mat.shape[0], self.word_embedding_dim, padding_idx=word_vec_mat.shape[0] - 1)
        self.word_embedding.weight.data.copy_(word_vec_mat)


    def forward(self, inputs):
        word = inputs['word']
        
        x = torch.cat([self.word_embedding(word)], 2)
        return x


