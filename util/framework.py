import os
import sklearn.metrics
import numpy as np
import sys,logging
import time
from . import sentence_encoder
from . import data_loader
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
# from pytorch_pretrained_bert import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score,accuracy_score
import pandas as pd

def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0

class FewShotEventModel(nn.Module):
    def __init__(self, my_sentence_encoder):
        '''
        sentence_encoder: Sentence encoder
        
        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        #self.sentence_encoder = nn.DataParallel(my_sentence_encoder)
        self.sentence_encoder = my_sentence_encoder
        self.cost = nn.CrossEntropyLoss()
        
    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        #print(logits)
        #print(logits.view(-1, N).shape,label.view(-1).shape)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

class FewShotEventFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, adv_data_loader=None, adv=False, d=None):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.adv_data_loader = adv_data_loader
        self.adv = adv 
        self.label_acc={}
        if adv:
            self.adv_cost = nn.CrossEntropyLoss()
            self.d = d
            self.d.cuda()
    
    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def train(self,
              model,
              model_name,
              B, N, K, Q,
              learning_rate=1e-1,
              train_iter=30000,
              val_iter=1000,
              val_step=2000,
              load_ckpt=None,
              save_ckpt=None,
              warmup_step=300,
              grad_iter=1,
              fp16=False,
              pair=False,
              model_grad=False
              ):
        '''
        model: a FewShotEventModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        '''
        print("Start training...")
    
        print('Use bert optim!')
        parameters_to_optimize = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize 
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in parameters_to_optimize
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 
      

        if load_ckpt:
            model_CKPT = torch.load(load_ckpt)
            model.load_state_dict(model_CKPT['state_dict'])
            optimizer.load_state_dict(model_CKPT['optimizer'])
            scheduler.load_state_dict(model_CKPT['scheduler'])

            start_iter = 0
        else:
            start_iter = 0

        

        model.train()

        # Training
        best_acc = 0
        iter_loss = 0.0
        iter_loss_dis = 0.0
        iter_right = 0.0
        iter_right_dis = 0.0
        iter_sample = 0.0
        for it in range(start_iter, start_iter + train_iter):
            if pair:
                batch, label,label_names = next(self.train_data_loader)
                if torch.cuda.is_available():
                    for k in batch:
                        batch[k] = batch[k].cuda()
                    label = label.cuda()
                logits, pred = model(batch, N, K, Q * N)
                #print('真实值：',label)
                #print('预测值：',pred)
            else:
                support, query, labels,label,label_names = next(iter(self.train_data_loader))
                if torch.cuda.is_available():
                    for k in support:
                        support[k] = support[k].cuda()
                        
                    for k in query:
                        query[k] = query[k].cuda()
                    for k in labels:
                        labels[k] = labels[k].cuda()
                    label = label.cuda()
                if model_grad:
                    loss, pred = model(support, query, labels,N, K, Q * N,label)
                else:
                    logits, pred = model(support, query, N, K, Q * N)
                    loss = model.loss(logits, label) / float(grad_iter)
            right = model.accuracy(pred, label)
            loss.backward()
            
            if it % grad_iter == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            iter_loss += self.item(loss.data)
            iter_right += self.item(right.data)
            iter_sample += 1
            if self.adv:
                sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%, dis_loss: {3:2.6f}, dis_acc: {4:2.6f}'
                    .format(it + 1, iter_loss / iter_sample, 
                        100 * iter_right / iter_sample,
                        iter_loss_dis / iter_sample,
                        100 * iter_right_dis / iter_sample) + '\r')
            else:
                sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample) + '\r')
            sys.stdout.flush()

            if (it + 1) % val_step == 0:
                acc = self.eval(model, B, N, K, Q, val_iter, pair=pair,model_grad=model_grad)
                model.train()
                if acc > best_acc:
                    print('Best checkpoint')
                    torch.save({'state_dict': model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()}, save_ckpt)
                    best_acc = acc
                iter_loss = 0.
                iter_loss_dis = 0.
                iter_right = 0.
                iter_right_dis = 0.
                iter_sample = 0.
                
        print("\n####################\n")
        print("Finish training " + model_name)

    def eval(self,
            model,
            B, N, K, Q,
            eval_iter,
            pair=False,
            ckpt=None,
            model_grad=False
            ): 
        '''
        model: a FewShotEventModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("###################################################")
        
        #model.eval()
        if ckpt is None:
            print("Use val dataset")
            eval_dataset = self.val_data_loader
        else:
            print("Use test dataset")
            if ckpt != 'none':
                model_CKPT = torch.load(ckpt)
                model.load_state_dict(model_CKPT['state_dict'])
            eval_dataset = self.test_data_loader

        iter_right = 0.0
        iter_sample = 0.0
        f1score=0
        for it in range(eval_iter):
            if model_grad:#maml类的
                support, query, labels,label, label_names = next(iter(eval_dataset))
                if torch.cuda.is_available():
                    for k in support:
                        #print(support[k].shape,type(support[k]))
                        support[k] = support[k].cuda()
                    for k in query:
                        query[k] = query[k].cuda()
                    for k in labels:
                        labels[k] = labels[k].cuda()
                    label = label.cuda()
                logits, pred = model(support, query, labels,N, K, Q * N,label)
                logits=logits.detach()
            else:
                if pair:
                    batch, label,label_names = next(eval_dataset)
                    if torch.cuda.is_available():
                        for k in batch:
                            batch[k] = batch[k].cuda()
                        
                        label = label.cuda()
                    logits, pred = model(batch, N, K, Q * N )
                    #print(label,pred)
                else:
                    support, query, labels,label,label_names = next(iter(eval_dataset))
                    if torch.cuda.is_available():
                        for k in support:
                            support[k] = support[k].cuda()
                        for k in query:
                            query[k] = query[k].cuda()
                        for k in labels:
                            labels[k] = labels[k].cuda()
                        label = label.cuda()
                    logits, pred = model(support, query, N, K, Q * N )

            right = model.accuracy(pred, label)
            iter_right += self.item(right.data)
               
            f1=f1_score(label.cpu().numpy(), pred.cpu().numpy(), average='macro')
            f1score+=f1

            iter_sample += 1
            sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f} | f1: {1:3.2f}'.format(it + 1, 100*iter_right / iter_sample, 100*(f1score.item() / iter_sample)) + '\r')
            logging.info('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) + '\r')
               
            logging.info('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100*(f1score.item() / iter_sample)) + '\r')
            sys.stdout.flush()
        return iter_right / iter_sample

    def test(self,
            model,
            N, K,
            eval_iter,
            ckpt=None,
            pair=False,
            num_class=10,
            model_grad=False
            ): 
        
        print("")
        
        #model.train()
        model_CKPT = torch.load(ckpt)
        model.load_state_dict(model_CKPT['state_dict'])
        eval_dataset = self.test_data_loader

        acc=0
        self.label_acc={}
        loss_list={}

        for it in range(eval_iter): 
            if model_grad:
                support, querys, labels,label_names= next(iter(eval_dataset))
                label_all=[]
                pred_all=[]
                for index in range(len(querys)):
                    query=querys[index]['data']
                    label=querys[index]['label']
                    Q_test=querys[index]['num']
                    if torch.cuda.is_available():
                        for k in support:
                            support[k] = support[k].cuda()
                        for k in query:
                            query[k] = query[k].cuda()
                        for k in labels:
                            labels[k] = labels[k].cuda()

                        label = label.cuda()
                    #print(Q_test,len(query['word']))
                    logits, pred = model(support, query,labels,N, K,Q_test,label)
                    label_all.append(label)
                    pred_all.append(pred)
            else:
                with torch.no_grad():
                    support, querys, labels,label_names= next(iter(eval_dataset))
                    label_all=[]
                    pred_all=[]
                    for index in range(len(querys)):
                        query=querys[index]['data']
                        label=querys[index]['label']
                        Q_test=querys[index]['num']
                        if torch.cuda.is_available():
                            for k in support:
                                support[k] = support[k].cuda()
                            for k in query:
                                query[k] = query[k].cuda()
                            label = label.cuda()
                        #print(Q_test,len(query['word']))
                        logits, pred = model(support, query, N, K,Q_test)
                        label_all.append(label)
                        pred_all.append(pred)
            label=torch.hstack(label_all)
            pred=torch.hstack(pred_all)
            n_c=len(pred)

                
            #集成测试只投赞同票和反对票
            if it%num_class==0:#初始化
                tmp_result=np.empty([num_class,n_c], dtype = int)
                test_label=label_names[0]
            tmp_result[it%num_class]=pred.cpu()

            if (it+1)%num_class==0:#计算完当前类别投票表决，并计算准确率
                tmp_result=tmp_result.T #(n_c,num_class)
                result=np.count_nonzero(tmp_result ==int(label[0]), axis=1)
                #print(result)
                loss_list[test_label]=np.where(result<num_class*0.8)
                self.label_acc[test_label]=np.sum(result > num_class//2)/n_c
                acc+=self.label_acc[test_label]
                print(f"{test_label} 的准确率是：{self.label_acc[test_label]*100}")
                print(f"{test_label} 的被打回的样本编号是：{loss_list[test_label]}")
                logging.info(f"{test_label} 的准确率是：{self.label_acc[test_label]*100}")
                    
                
        print(self.label_acc)
        acc/=len(self.label_acc)
        logging.info(f"在整个测试数据集上的准确率是{acc*100}")
        print(f"在整个测试数据集上的准确率是{acc*100}")
        #logging.info(self.label_acc)
        return self.label_acc,acc
    
    def get_task(self,
              model,
              N, K, Q,
              load_ckpt=None,
              sim_type='GC'
              ):
        '''
        model: a FewShotEventModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        ''' 
        print("get task...")
        model_CKPT = torch.load(load_ckpt)
        model.load_state_dict(model_CKPT['state_dict'])

        #print(self.train_data_loader.dataset[3][0]['word'][0])
        
            

        model.eval()
        num_train_task=1000
        num_test_task=100
        params=[]
         
        for i in range(num_train_task):
            support, query, labels,label,label_names = self.train_data_loader.dataset[i]
            if torch.cuda.is_available():
                for k in support:
                    support[k] = torch.stack(support[k]).cuda()
                    #print(support[k])
                for k in query:
                    query[k] =torch.stack(query[k]).cuda()
                for k in labels:
                    labels[k] = torch.stack(labels[k]).cuda()
                label = torch.tensor(label).cuda()
            grad= model.get_params_change(support, labels,N, K,sim_type)
            #print(grad[-10:])
            #print(grad.shape)
            params.append(grad.detach())
        params=torch.stack(params).detach()
        print(params.shape)
        

        if sim_type!='GC':
            params=params.unsqueeze(1)#(num,1,n,d)

        task=np.zeros((num_test_task,num_train_task))
        for i in range(num_test_task):
            support, query, labels,label,label_names = self.test_data_loader.dataset[i]
            #print(label_names)
            if torch.cuda.is_available():
                for k in support:
                    support[k] = torch.stack(support[k]).cuda()
                    #print(support[k])
                for k in query:
                    query[k] =torch.stack(query[k]).cuda()
                for k in labels:
                    labels[k] = torch.stack(labels[k]).cuda()
                label = torch.tensor(label).cuda()
            
            params_test= model.get_params_change(support, labels,N, K,sim_type).detach()#N,D
            if sim_type=='GC':
                result=torch.cosine_similarity(params,params_test,-1)
            else:
                #print(params.shape)
                params_test=params_test.unsqueeze(1)#N,1,D
                #print(params_test.shape)
                if sim_type=='sim_ou':
                    result=-(torch.pow(params-params_test, 2)).sum(-1)
                    result,_=result.max(-1)
                    result=result.sum(-1).squeeze(0)
                if sim_type=='sim_cos':
                    result=torch.cosine_similarity(params,params_test,-1)
                    result,_=result.max(-1)
                    result=result.sum(-1).squeeze(0)
                elif sim_type=='sim_dot':
                    result=(params*params_test).sum(-1)
                    result,_=result.max(-1)
                    result=result.sum(-1).squeeze(0)
            

            tmp=torch.argsort(result).cpu().numpy()#从小到大排
            tmp=tmp[::-1]#从大到小排
            task[i]=tmp
            #print(tmp)
            
        
        pd.DataFrame(task).to_csv(f"task_{sim_type}.csv")
    
            

    def retrain(self,model,N, K, Q,
              learning_rate=1e-1,
              load_ckpt=None,
              warmup_step=300,
              sim_type='GC',
              train_iter=10
              ):
       
        print("Start training...")
        #train_iter=5
    
        
        parameters_to_optimize = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize 
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in parameters_to_optimize
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 
        

        
        model_CKPT = torch.load(load_ckpt)
        
            
        
        model.train()
        num_test_task=100
        task=pd.read_csv(f"task_{sim_type}.csv",index_col=0)
        a_loss,a_acc,b_loss,b_acc=0,0,0,0
        loss_task=[]


        for i in range(num_test_task):

            model.load_state_dict(model_CKPT['state_dict'])
            optimizer.load_state_dict(model_CKPT['optimizer'])
            scheduler.load_state_dict(model_CKPT['scheduler'])

            support, query, labels,label,label_names = self.test_data_loader.dataset[i]
            if torch.cuda.is_available():
                for k in support:
                    support[k] = torch.stack(support[k]).cuda()
                for k in query:
                    query[k] =torch.stack(query[k]).cuda()
                for k in labels:
                    labels[k] = torch.stack(labels[k]).cuda()
                label = torch.tensor(label).cuda()
               
            bloss, pred = model(support, query, labels,N, K, Q * N,label)
            bloss=bloss.detach()    
            bacc = model.accuracy(pred, label)

            b_loss+=bloss.item()
            b_acc+=bacc.item()

            tmp=task.iloc[i,0:train_iter]
            for id in tmp:
                support_train, query_train, labels_train,label_train,label_names_train = self.train_data_loader.dataset[int(id)]
                for k in support:
                    support_train[k] = torch.stack(support_train[k]).cuda()
                for k in query:
                    query_train[k] =torch.stack(query_train[k]).cuda()
                for k in labels:
                    labels_train[k] = torch.stack(labels_train[k]).cuda()
                label_train = torch.tensor(label_train).cuda()
                losstmp, predtmp = model(support_train, query_train, labels_train,N, K, Q * N,label_train) 
                losstmp.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            #二次训练后重新测试
            aloss, apred = model(support, query, labels,N, K, Q * N,label)    
            aloss=aloss.detach()   
            aacc = model.accuracy(apred, label)

            a_loss+=aloss.item()
            a_acc+=aacc.item()
            
            if aloss<bloss:
                print(i,bloss.item(),aloss.item(),bacc.item(),aacc.item(),'***')
                logging.info(f'任务{i}：loss：{bloss.item()}，{aloss.item()}，acc:{bacc.item()},{aacc.item()}****')
            else:
                print(i,bloss.item(),aloss.item(),bacc.item(),aacc.item())
                logging.info(f'任务{i}：loss：{bloss.item()}，{aloss.item()}，acc:{bacc.item()},{aacc.item()}')
                loss_task.append(i)

        print(f'不经过二次训练所有任务均值 loss:{b_loss/(i+1)}，准确率{b_acc/(i+1)}')
        logging.info(f'不经过二次训练所有任务均值 loss:{b_loss/(i+1)}，准确率{b_acc/(i+1)}')
        
        print(f'经过二次训练所有任务均值 loss:{a_loss/(i+1)}，准确率{a_acc/(i+1)}')
        logging.info(f'经过二次训练所有任务均值 loss:{a_loss/(i+1)}，准确率{a_acc/(i+1)}')
        print(len(loss_task),loss_task)
        


       
        

            
    