import sys
sys.path.append('..')
import util
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class Pair(util.framework.FewShotEventModel):
    
    def __init__(self, sentence_encoder, hidden_size=230):
        util.framework.FewShotEventModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()

    def forward(self, batch, N, K, total_Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        #将每一个query data和每一个support data拼接在一起，（如果是5-5-1，5-way-5-shot-1-q，每个类一个query所以总的是5*5*5=125组
        将这些拼接好的数据送到bert中得到一个分类的分数，如果batch是4，那么是(500,2)
        然后针对每一类做一个平均，得到q和每一类数据的得分，然后选出得分最高的作为目标类别
        '''
        '''
        logits = self.sentence_encoder(batch) #(5-5-1:(500,2))
        #print(logits.shape)
        logits = logits.view(-1, total_Q, N, K, 2)#(4,5,5,5,2)
        #print(logits.shape)
        logits = logits.mean(3) # (-1, total_Q, N, 2)#(4,5,5,2)
        #print(logits.shape)
        logits_na, _ = logits[:, :, :, 0].min(2, keepdim=True) # (-1, totalQ, 1)
        logits = logits[:, :, :, 1] # (-1, total_Q, N),(4,5,5)
        #print(logits.shape)
        logits = torch.cat([logits, logits_na], 2) # (B, total_Q, N + 1),(4,5,6)
        #print(logits.shape)
        _, pred = torch.max(logits.view(-1, N + 1), 1)
        #print(_,pred)
        
        '''
        logits = self.sentence_encoder(batch)
        #print(logits)
        logits=F.relu(logits)
        logits = logits.view(-1, total_Q, N, K, 2)
        logits = logits.mean(3)#(-1, total_Q, N, 2)
        logits_na=logits[:, :, :, 0].sum(2)
        logits_na=logits_na.view(-1,total_Q, 1)
        logits=logits[:,:,:,1]-logits[:,:,:,0]+logits_na
        logits = torch.cat([logits, logits_na], 2)
        _, pred = torch.max(logits.view(-1, N + 1), 1)
        
        
        return logits, pred
