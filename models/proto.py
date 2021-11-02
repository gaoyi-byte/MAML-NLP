import sys

from torch._C import LongStorageBase
sys.path.append('..')
import util
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class Proto(util.framework.FewShotEventModel):
    
    def __init__(self, sentence_encoder, dot=False,cos=False):
        util.framework.FewShotEventModel.__init__(self, sentence_encoder)
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        #self.classifier = nn.Linear()
        self.dot = dot
        self.cos = cos
        #self.W = torch.eye(support.shape[-1])

    def __dist__(self, x, y, dim1):
        if self.dot:
            return (x * y).sum(dim1)
        elif self.cos:
            return torch.cosine_similarity(x, y,dim=dim1)   
        else:
            return -(torch.pow(x - y, 2)).sum(dim1)

    def __batch_dist__(self, S, Q):
        #print(S.shape,Q.shape)
        
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, support, query, N, K, total_Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        support_emb = self.sentence_encoder(support) # (B * N * K, D), where D is the hidden size
        query_emb = self.sentence_encoder(query) # (B * total_Q, D)
        hidden_size = support_emb.size(-1)
        support = self.drop(support_emb)
        query = self.drop(query_emb)
        support = support.view(-1, N, K, hidden_size) # (B, N, K, D)
        query = query.view(-1, total_Q, hidden_size) # (B, total_Q, D)

        # Prototypical Networks 
        # Ignore NA policy
        support = torch.mean(support, 2) # Calculate prototype for each class
        #support=support.dot(self.W)
        #query=query.dot(self.W)
        logits = self.__batch_dist__(support, query) # (B, total_Q, N)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2) # (B, total_Q, N + 1) 
        _, pred = torch.max(logits.view(-1, N + 1), 1)
        #看看怎么计算loss
        #print(pred)
        return logits, pred

        
    
    
