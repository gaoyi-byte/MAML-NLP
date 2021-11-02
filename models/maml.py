import sys

from torch.nn.modules import batchnorm
sys.path.append('..')
import util
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from copy import deepcopy

class Learner(nn.Module):

    def __init__(self,sentence_encoder,hidden_size,N):
        nn.Module.__init__(self)
        self.sentence_encoder=sentence_encoder
        self.linear= nn.Linear(hidden_size,hidden_size,bias=False)
        '''
        for name in self.sentence_encoder.state_dict():
            print(name,self.sentence_encoder.state_dict()[name].shape)
        for name in self.linear.state_dict():
            print(name,self.linear.state_dict()[name].shape)
        '''
        
        
    def change_params(self,vars):

        k=0
        for param in self.sentence_encoder.parameters():
            param.data.copy_(vars[k])
            k+=1
        for param in self.linear.parameters():
            param.data.copy_(vars[k])
            k+=1


    def forward(self, inputs,labels):

       
        support_emb = self.sentence_encoder(inputs)# B*N*K,D
        label_emb=self.sentence_encoder(labels)# B*N,D 
        '''
        emb=[]
        for i in support_emb:
            c=torch.cat((i.repeat(label_emb.shape[0],1),label_emb),1)
            emb.append(c)
        emb=torch.stack(emb)#(b*n*k,n,d)
        '''

        logits = self.linear(support_emb)

        #print(logits.shape,label_emb.T.shape)
        logits=torch.mm(logits,label_emb.T)
            

        return logits,support_emb,label_emb
        
        

class MAML(util.framework.FewShotEventModel):
    
    def __init__(self, sentence_encoder, N, K, hidden_size=230,update_lr=2e-5):  
        '''
        N: num of classes
        K: num of instances for each class in the support set
        '''
        util.framework.FewShotEventModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        #self.drop = nn.Dropout()
        self.learner=Learner(sentence_encoder,hidden_size,N)
        self.update_lr=update_lr
        self.update_step=5
        self.test_step=5
        self.meta_optim = optim.Adam(self.learner.parameters(), lr=self.update_lr)
    

    def forward(self, support, query, labels,N, K, totalQ,label):
        
        vars= deepcopy(self.learner)#保存原来的参数

        loss_tmp=[]
        for k in range(0, self.test_step):#内层更新多步
            logits,_1,_2 = self.learner(support,labels)#support结果
            logits = logits.view(-1, N, K, N) # (B, N, K, N)
            #print(logits)
            B=logits.shape[0]
            #self.zero_grad() 
            tmp_label = Variable(torch.tensor([[x] * K for x in range(N)] * B, dtype=torch.long).cuda())
            loss = self.cost(logits.view(-1, N), tmp_label.view(-1))#根据support更新数据，但不进行反向传播，只更新得到临时参数θ
            self.meta_optim.zero_grad()
            loss.backward()
            self.meta_optim.step()
            

            logits_q,_1,_2 = self.learner(query,labels)
            loss_q=self.cost(logits_q.view(-1, N), label.view(-1))
            loss_tmp.append(loss_q.item())

            
        self.learner.change_params(list(vars.parameters()))
        _, pred = torch.max(logits_q.view(-1, N), 1)
        #print(loss_tmp)
       
        
        return loss_q,pred



    def get_params_change(self, support, labels,N, K,sim_type):
        
        vars= deepcopy(self.learner)#保存原来的参数

        for k in range(0, self.test_step):#内层更新多步
            logits,support_emb,label_emb = self.learner(support,labels)#support结果
            logits = logits.view(-1, N, K, N) # (B, N, K, N)
            #print(logits)
            B=logits.shape[0]
            #self.zero_grad() 
            tmp_label = Variable(torch.tensor([[x] * K for x in range(N)] * B, dtype=torch.long).cuda())
            loss = self.cost(logits.view(-1, N), tmp_label.view(-1))#根据support更新数据，但不进行反向传播，只更新得到临时参数θ
            #self.meta_optim.zero_grad()
            loss.backward()
            self.meta_optim.step()
        
        if sim_type=='GC':
            with torch.no_grad():
                '''
                参数量太太太大了，不ok
                result=[]
                print(len(list(self.learner.parameters())))
                for i in range(len(list(self.learner.parameters()))):
                    result.append(torch.flatten(list(self.learner.parameters())[i]-list(vars.parameters())[i]))
                result=torch.hstack(result)
                '''
                result=torch.flatten(list(self.learner.parameters())[-1]-list(vars.parameters())[-1])
                #print(result.shape)
                #print(result.sum().detach())
            self.learner.change_params(list(vars.parameters()))
            return result
        else:
            support_emb=support_emb.reshape(N,K,-1).mean(1)
            self.learner.change_params(list(vars.parameters()))
            return support_emb
       
        
        

    

