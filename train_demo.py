from util.data_loader import get_loader, get_loader_test
from util.framework import FewShotEventFramework
from util.sentence_encoder import CNNSentenceEncoder, BERTSentenceEncoder, BERTPAIRSentenceEncoder, RobertaSentenceEncoder, RobertaPAIRSentenceEncoder
import models
from models.proto import Proto
from models.gnn import GNN
from models.snail import SNAIL
from models.metanet import MetaNet
from models.siamese import Siamese
from models.pair import Pair
from models.d import Discriminator
from models.mtb import Mtb
from models.maml import MAML
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import os
import logging
import os
import random

 
def main():
    logger = logging.getLogger(__name__)
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename="my.log")
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='few_shot_train.json',
                        help='train file')
    parser.add_argument('--val', default='few_shot_dev.json',help='val file') 
    parser.add_argument(
        '--test', default='few_shot_test.json', help='test file')
    parser.add_argument('--adv', default=None,help='adv file')
    parser.add_argument('--N', default=5, type=int,
                        help='N way')
    parser.add_argument('--K', default=5, type=int,
                        help='K shot')
    parser.add_argument('--Q', default=15, type=int,
                        help='Num of query per class')
    parser.add_argument('--Q_test', default=50, type=int,
                        help='Num of query per class in test')
    parser.add_argument('--num_class', default=20, type=int,
                        help='Num of query class per task')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch size')
    parser.add_argument('--train_iter', default=10000, type=int,
                        help='num of iters in training')
    parser.add_argument('--train_num', default=10, type=int,
                        help='num of train task in retrain')
    parser.add_argument('--val_iter', default=1000, type=int,
                        help='num of iters in validation')
    parser.add_argument('--test_iter', default=1000, type=int,
                        help='num of iters in testing')
    parser.add_argument('--val_step', default=1000, type=int,
                        help='val after training how many iters')
    parser.add_argument('--model', default='proto',
                        help='model name')
    parser.add_argument('--mode', default='train',
                        help='retrain/get_task')
    parser.add_argument('--sim_type', default='GC',
                        help='sim_cos/sim_ou')

    parser.add_argument('--seed', default=666, type=int,
                        help='random seed')
    parser.add_argument('--encoder', default='bert',
                        help='encoder: cnn or bert or roberta')
    parser.add_argument('--bert_type', default='fin',
                        help='en,ch,fin')
    parser.add_argument('--max_length', default=192, type=int,
                        help='max length')
    parser.add_argument('--lr', default=-1, type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='weight decay')
    parser.add_argument('--dropout', default=0.0, type=float,
                        help='dropout rate')
    parser.add_argument('--grad_iter', default=1, type=int,
                        help='accumulate gradient every x iterations')
    parser.add_argument('--optim', default='sgd',
                        help='sgd / adam / adamw')
    parser.add_argument('--hidden_size', default=768, type=int,
                        help='hidden size')
    parser.add_argument('--load_ckpt', default=None,
                        help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
                        help='save ckpt')
    parser.add_argument('--fp16', action='store_true',
                        help='use nvidia apex fp16')
    parser.add_argument('--only_test', action='store_true',
                        help='only test')
    parser.add_argument('--eng', action='store_true',
                        help='only test')
    parser.add_argument('--ckpt_name', type=str, default='',
                        help='checkpoint name.')

    # only for bert / roberta
    parser.add_argument('--pair', action='store_true',
                        help='use pair model')

    # only for prototypical networks
    parser.add_argument('--dot', action='store_true',
                        help='use dot instead of L2 distance for proto')
    parser.add_argument('--cos', action='store_true',
                        help='use cos instead of L2 distance for proto')

    # only for mtb
    parser.add_argument('--no_dropout', action='store_true',
                        help='do not use dropout after BERT (still has dropout in BERT).')

    # experiment
    parser.add_argument('--use_sgd_for_bert', action='store_true',
                        help='use SGD instead of AdamW for BERT.')
    
    

    opt = parser.parse_args()
    if opt.eng:
        opt.bert_type='en'
        opt.val=opt.train
        opt.test=opt.train
    print(opt)
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    max_length = opt.max_length
    bert_type = opt.bert_type
    model_grad=False

    #设计随机种子
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.backends.cudnn.deterministic=True


    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))


    if encoder_name == 'cnn':
        try:
            glove_mat = np.load('./pretrain/glove/glove_mat.npy')
            glove_word2id = json.load(
                open('./pretrain/glove/glove_word2id.json'))
        except:
            raise Exception(
                "Cannot find glove files. Run glove/download_glove.sh to download glove files.")
        sentence_encoder = CNNSentenceEncoder(
            glove_mat,
            glove_word2id,
            max_length)

    elif encoder_name == 'bert':
        if bert_type == 'fin':
            print('使用finbert')
            pretrain_ckpt = 'pretrain/FinBERT'
            if opt.pair:
                sentence_encoder = BERTPAIRSentenceEncoder(pretrain_ckpt, max_length)
            else:
                sentence_encoder = BERTSentenceEncoder(pretrain_ckpt, max_length)
        elif bert_type == 'ch':
            print('使用中文的bert')
            pretrain_ckpt = 'bert-base-chinese'
            sentence_encoder = BERTSentenceEncoder( pretrain_ckpt, max_length)
        else:
            print('使用英文的bert')
            pretrain_ckpt ='bert-base-uncased'
            if opt.pair:
                sentence_encoder = BERTPAIRSentenceEncoder(pretrain_ckpt,max_length)
            else:
                sentence_encoder = BERTSentenceEncoder(pretrain_ckpt,max_length)

    else:
        raise NotImplementedError

    print('读取训练文件')
    logger.info("***** training *****")
    
    train_data_loader = get_loader('train', opt.train, opt.eng,sentence_encoder,
                                       N=N, K=K, Q=Q, batch_size=batch_size)
    val_data_loader = get_loader('val', opt.val, opt.eng,sentence_encoder,
                                     N=N, K=K, Q=Q, batch_size=batch_size)
    test_data_loader = get_loader('test', opt.test, opt.eng,sentence_encoder,
                                     N=N, K=K, Q=Q, batch_size=batch_size)
    
    # test_data_loader,test_iter = get_loader_test('test', opt.test, sentence_encoder,
    #                                  N=N, K=K, Q_test=opt.Q_test, num_class=opt.num_class)
    

   # 选择优化器
    if opt.optim == 'sgd':
        pytorch_optim = optim.SGD
    elif opt.optim == 'adam':
        pytorch_optim = optim.Adam
    elif opt.optim == 'adamw':
        from transformers import AdamW
        pytorch_optim = AdamW
    else:
        raise NotImplementedError
    if opt.adv:
        d = Discriminator(opt.hidden_size)
        framework = FewShotEventFramework(
            train_data_loader, val_data_loader, test_data_loader, adv=opt.adv, d=d)
    else:
        framework = FewShotEventFramework(
            train_data_loader, val_data_loader, test_data_loader)

    prefix = '-'.join([model_name, bert_type, encoder_name, str(N), str(K)])
    if opt.adv is not None:
        prefix += '-adv_' + opt.adv
    if opt.dot:
        prefix += '-dot'
    if opt.cos:
        prefix += '-cos'
    if len(opt.ckpt_name) > 0:
        prefix += '-' + opt.ckpt_name
    print(prefix)
    
    if encoder_name in ['bert', 'roberta']:
        bert_optim = True
    else:
        bert_optim = False
    if opt.lr == -1:
        if bert_optim:
            # opt.lr = 2e-5 #原来的     
            opt.lr = 2e-5
        else:
            opt.lr = 1e-1

    if model_name == 'proto':
        model = Proto(sentence_encoder, dot=opt.dot, cos=opt.cos)
    elif model_name == 'gnn':
        model = GNN(sentence_encoder, N, hidden_size=opt.hidden_size)
    elif model_name == 'proto_norm':
        model = GNN(sentence_encoder, N, hidden_size=opt.hidden_size)
    elif model_name == 'snail':
        model = SNAIL(sentence_encoder, N, K, hidden_size=opt.hidden_size)
    elif model_name == 'maml':
        model = MAML(sentence_encoder, N, K, hidden_size=opt.hidden_size,update_lr=opt.lr)
        model_grad=True
    elif model_name == 'siamese':
        model = Siamese(sentence_encoder,
                        hidden_size=opt.hidden_size, dropout=opt.dropout)
    elif model_name == 'pair':
        model = Pair(sentence_encoder, hidden_size=opt.hidden_size)
    elif model_name == 'mtb':
        model = Mtb(sentence_encoder, use_dropout=not opt.no_dropout)
    else:
        raise NotImplementedError

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test and opt.mode=='train':
        framework.train(model, prefix, batch_size, N, K, Q, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                         val_step=opt.val_step, pair=opt.pair,
                        train_iter=opt.train_iter, val_iter=opt.val_iter, 
                        learning_rate=opt.lr, model_grad=model_grad)
    elif opt.mode=='retrain':
        framework.retrain(model, N, K, Q, load_ckpt=opt.load_ckpt,
                        learning_rate=opt.lr,sim_type=opt.sim_type,train_iter=opt.train_num)
    elif opt.mode=='get_task':
        framework.get_task(model, N, K, Q,load_ckpt=opt.load_ckpt,sim_type=opt.sim_type)
    else:
        ckpt = opt.load_ckpt
        acc = framework.eval(model, batch_size,N, K, Q,opt.val_iter,ckpt=ckpt, model_grad=model_grad)
        print("RESULT: %.2f" % (acc * 100))
        logger.info("RESULT: %.2f" % (acc * 100))

    #acc = framework.test(model, N, K, test_iter,ckpt=ckpt, pair=opt.pair,num_class=opt.num_class,model_grad=model_grad)
    


if __name__ == "__main__":
    cpu_num = 1  # 这里设置成你想运行的CPU个数
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    main()
