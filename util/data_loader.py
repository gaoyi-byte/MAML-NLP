import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
import itertools



class FewEventDataset(data.Dataset):
    """
    FewEvent Dataset
    """
    def __init__(self,types, name, eng,encoder, N, K, Q,  root):
        self.root = root
        path = os.path.join(root, name)
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.encoder = encoder
        #self.encoder_name = encoder_name
        self.max_length = encoder.max_length
        self.types=types
        self.seq=list(itertools.combinations(range(len(self.classes)), self.N))
        np.random.shuffle(self.seq)
        print(len(self.seq))
        if eng:
            if types=='train':
                self.classes=self.classes[:80]
            elif types=='val':
                self.classes=self.classes[80:90]
            else:
                self.classes=self.classes[90:]
        
        print(self.types+'总共的类别数：',len(self.classes))
        print('类别：',self.classes)

    def __additem__(self, d, word, mask):
        d['word'].append(word)
        d['mask'].append(mask)
    
    def __getitem__(self, index):
        
        #target_classes = random.sample(self.classes, self.N)
        target_classes = [self.classes[i] for i in self.seq[index%len(self.seq)]]
        #print(index,target_classes)
        support_set = {'word': [],  'mask': [] }
        query_set = {'word': [],  'mask': [] }
        label_set = {'word': [],  'mask': [] }
        query_label = []
        #print(target_classes)

        for i, class_name in enumerate(target_classes):
            #label embedding
            word,  mask = self.encoder.tokenize(class_name)
            word = torch.tensor(word).long()
            mask = torch.tensor(mask).long()
            self.__additem__(label_set, word,  mask)

            #处理query和support
            #indices = np.random.choice(list(range(len(self.json_data[class_name]))), self.K + self.Q, False)
            indices=range(self.K + self.Q)
            #print(indices)
            count = 0
            for j in indices:
                word,  mask = self.encoder.tokenize(self.json_data[class_name][j][0])
                word = torch.tensor(word).long()
                mask = torch.tensor(mask).long()
                if count < self.K:
                    self.__additem__(support_set, word,  mask)
                else:
                    self.__additem__(query_set, word, mask)
                count += 1

            query_label += [i] * self.Q

        return support_set, query_set, label_set, query_label, target_classes
    
    def __len__(self): 
        return 1000000000

def collate_fn(data):
    batch_support = {'word': [],  'mask': []}
    batch_query = {'word': [],  'mask': []}
    batch_labels={'word': [],  'mask': []}
    batch_label = []
    batch_label_name = []
    support_sets, query_sets, label_sets,query_labels,label_names = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        for k in label_sets[i]:
            batch_labels[k]+=label_sets[i][k]
        batch_label += query_labels[i]
        batch_label_name += label_names[i]

    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    for k in batch_labels:
        batch_labels[k] = torch.stack(batch_labels[k], 0)

    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_labels,batch_label,batch_label_name

def get_loader(types,name,eng, encoder, N, K, Q, batch_size, 
        num_workers=1, collate_fn=collate_fn,root='./data'):
    dataset = FewEventDataset(types,name, eng,encoder, N, K, Q,root)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    print(len(data_loader.dataset))
    #return iter(data_loader)
    return data_loader

class FewEventtest(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self,types, name, encoder, N, K,root,Q_test,num_class):
        self.root = root
        path = os.path.join(root, name)
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())[:10]
        self.N = N
        self.K = K
        self.encoder = encoder
        #self.encoder_name = encoder_name
        self.max_length = encoder.max_length
        self.types=types
        self.seq=[]
        self.Q_test=Q_test
        self.seq=self.creat_batch(len(self.classes),num_class)
        
        
        
        print(self.types+'总共的类别数：',len(self.classes))
        print('类别：',self.classes)

    def creat_batch(self,n,num_class):
        #count=np.zeros(n)
        seq=[]
        '''
        i=8
        list_i=list(range(0,i))+list(range(i+1,n))
        seq_i=list(itertools.combinations(list_i, self.N-1))
        for j in range(len(seq_i)):
            seq.append(tuple([i])+seq_i[j])
        '''
        for i in range(0,n):
            #print(i,count)
            list_i=list(range(0,i))+list(range(i+1,n))
            seq_i=list(itertools.combinations(list_i, self.N-1))
            indices = np.random.choice(range(len(seq_i)),num_class, False)
            for j in indices:
                seq.append(tuple([i])+seq_i[j])
       
        return seq

    def __additem__(self, d, word, mask,seg=None):
        d['word'].append(word)
        d['mask'].append(mask)
        #d['seg'].append(seg)
 

    def __getitem__(self, index):#每次只检测一个类，只取一个类的query数据  
        num_iter=len(self.seq)
        target_classes = [self.classes[i] for i in self.seq[index%num_iter]]
        support_set = {'word': [], 'mask': []} 
        query_set = {'word': [], 'mask': []}
        label_set = {'word': [], 'mask': []}
        query=[]
        #读取support data
        
            
        for i, class_name in enumerate(target_classes):
            word, mask = self.encoder.tokenize(class_name)
            word = torch.tensor(word).long()  
            mask = torch.tensor(mask).long()
            self.__additem__(label_set, word,  mask)

            for j in range(self.K):
                word, mask = self.encoder.tokenize(self.json_data[class_name][j][0])

                word = torch.tensor(word).long()  
                mask = torch.tensor(mask).long()

                self.__additem__(support_set, word,  mask)
       
        #在这里分批次处理query data，只取第一个类的所有数据
        start=self.K 
        query_name=target_classes[0]
        while start<len(self.json_data[query_name]):
            tmp={}
            end=start+self.Q_test
            query_set = {'word': [], 'mask': []}
            query_label = []
            for j in range(start,end):
                if j>=len(self.json_data[query_name]):
                        j=j-1
                        break
                word, mask = self.encoder.tokenize(self.json_data[query_name][j][0])
                word = torch.tensor(word).long()
                mask = torch.tensor(mask).long()
                #seg = torch.tensor(seg).long()
                self.__additem__(query_set, word,  mask)

            j+=1
            query_label += [0] * (j-start)
            tmp['data']=query_set
            tmp['label']=query_label
            tmp['num']=j-start
            #print(len(query_label))
            query.append(tmp)
            start=end

        return support_set, query, label_set, target_classes
    
    def __len__(self): 
        return 1000000000

def collate_fn_test(data):
    query=[]
    batch_support = {'word': [],  'mask': []}
    batch_label_set = {'word': [],  'mask': []}
    support_sets, querys,labels,label_names= zip(*data)
    #print(querys)
    #处理support数据
    for i in range(len(support_sets)):
        #print(i,len(support_sets))
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in labels[i]:
            batch_label_set[k] += labels[i][k]
        

    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_label_set:
        batch_label_set[k] = torch.stack(batch_label_set[k], 0)

    #处理query数据
    for i in range(len(querys)):
        for n in range(len(querys[i])):#里面是每一个batch,里面是data和label
            #print(querys[i][n])
            tmp={}
            batch_query = {'word': [],  'mask': []}
            batch_label = []
            for k in querys[i][n]['data']:
                batch_query[k] += querys[i][n]['data'][k]
            for k in batch_query:
                batch_query[k] = torch.stack(batch_query[k], 0)
            batch_label += querys[i][n]['label']
            batch_label = torch.tensor(batch_label)
            tmp['data']=batch_query
            tmp['label']=batch_label
            tmp['num']=querys[i][n]['num']
            query.append(tmp)
    return batch_support, query,batch_label_set,label_names[0]

def get_loader_test(types,name, encoder,N, K, Q_test, num_class,
        num_workers=1, collate_fn=collate_fn_test, root='./data'):
    dataset = FewEventtest(types,name, encoder,N, K, root,Q_test,num_class)
    data_loader = data.DataLoader(dataset=dataset,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader),len(dataset.seq)
 


class FewEventUnsupervisedDataset(data.Dataset):
    """
    FewEvent Unsupervised Dataset
    """
    def __init__(self, name, encoder, N, K, Q, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.N = N
        self.K = K
        self.Q = Q
        self.encoder = encoder

    def __getraw__(self, item):
        word,mask = self.encoder.tokenize(item[0])
        return word, mask 

    def __additem__(self, d, word,  mask):
        d['word'].append(word)
        d['mask'].append(mask)

    def __getitem__(self, index):
        total = self.N * self.K
        support_set = {'word': [], 'mask': [] }

        indices = np.random.choice(list(range(len(self.json_data))), total, False)
        for j in indices:
            word, mask = self.__getraw__(
                    self.json_data[j])
            word = torch.tensor(word).long()
            mask = torch.tensor(mask).long()
            self.__additem__(support_set, word, mask)

        return support_set
    
    def __len__(self):
        return 1000000000

def collate_fn_unsupervised(data):
    batch_support = {'word': [], 'mask': []}
    support_sets = data
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    return batch_support

def get_loader_unsupervised(name, encoder, N, K, Q, batch_size, 
        num_workers=1, collate_fn=collate_fn_unsupervised, root='./data'):
    dataset = FewEventUnsupervisedDataset(name, encoder, N, K, Q,  root)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    #return iter(data_loader)
    return data_loader



