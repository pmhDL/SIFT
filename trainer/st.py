""" SIFT method """
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler
from models.SIFT import Learner
from utils.misc import Averager, Timer, count_acc, compute_confidence_interval, get_cos_similar_matrix
from dataloader.dataset_loader import DatasetLoader as Dataset

class ST(object):
    def __init__(self, args):
        self.args = args
        self.model = Learner(self.args, mode='st')
        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()

    def sample_base_data1(self, feat_base, label_base, baselabels, proto, num):
        feat_b = []
        label_b = []
        abs_lb = []
        c = 0
        for labell in baselabels:
            id = np.where(label_base == labell)[0]
            if self.args.selectm == 'nn2basecenter':
                feat_center = feat_base[id].mean(0)
                dist = -((feat_base[id]-feat_center)**2).sum(1)
                _, idx = torch.topk(torch.tensor(dist), num)
                idd = id[idx]
            elif self.args.selectm == 'nn2suppcenter':
                dist = -((feat_base[id] - proto[c]) ** 2).sum(1)
                _, idx = torch.topk(torch.tensor(dist), num)
                idd = id[idx]
            elif self.args.selectm == 'randomselect':
                idd = random.sample(list(id), num)
            feat_b.append(feat_base[idd])
            label_b.extend([c]*num)
            abs_lb.extend([labell]*num)
            c = c+1
        feat_b = np.concatenate(feat_b, axis=0)
        label_b = np.array(label_b)
        abs_lb = np.array(abs_lb)
        return feat_b, label_b, abs_lb

    def eval(self):
        """The function for the meta-eval phase."""
        # Load test set
        test_set = Dataset('test', self.args)
        sampler = CategoriesSampler(test_set.label, 600, self.args.way, self.args.shot + self.args.val_query)
        loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
        # Set test accuracy recorder
        test_acc_record = np.zeros((600,))
        # Set accuracy averager
        ave_acc = Averager()

        '''--------------- Generate labels --------------'''
        # Generate labels
        label = torch.arange(self.args.way).repeat(self.args.val_query)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        if torch.cuda.is_available():
            label_shot = label_shot.type(torch.cuda.LongTensor)
        else:
            label_shot = label_shot.type(torch.LongTensor)

        '''------------------- load data ------------------'''
        path = self.args.dataset_dir
        traindata = np.load(path+'/feat-train.npz')
        featbase, labelbase = traindata['features'], traindata['targets']
        train_class = len(np.unique(labelbase))
        trainemb = np.load(path + '/few-shot-wordemb-train.npz')['features']

        emb_novel = np.load(path + '/few-shot-wordemb-test.npz')['features']
        feat_base = featbase
        label_base = labelbase
        emb_base = trainemb

        sim_n_b = get_cos_similar_matrix(emb_novel, emb_base)
        maxv, maxid = sim_n_b.max(1), sim_n_b.argmax(1)

        timer = Timer()
        for i, batch in enumerate(loader, 1):
            label1 = label
            label1 = label1.cuda().data.cpu().numpy()
            if torch.cuda.is_available():
                data, label_abs = [_.cuda() for _ in batch]
            else:
                data, label_abs = batch[0], batch[1]
            k = self.args.way * self.args.shot
            data_shot, data_query = data[:k], data[k:]
            data_shot = F.normalize(data_shot, dim=1)
            data_query = F.normalize(data_query, dim=1)

            '''----------------- generate samples ------------------'''
            lb_abs = label_abs[: self.args.way]
            # the corresponding base classes
            lb_abs = lb_abs.cuda().data.cpu().numpy()
            base_lb_abs = maxid[lb_abs]

            support = data_shot.view(self.args.shot, self.args.way, -1).transpose(1, 0)
            proto = torch.mean(support, dim=1)
            proto = proto.cuda().data.cpu().numpy()
            feat_b, label_b, abs_lb_b = self.sample_base_data1(feat_base, label_base, base_lb_abs, proto, self.args.num_aug)
            sem_b = emb_base[abs_lb_b]
            sem_b1 = emb_base[base_lb_abs]

            feat_ns = data_shot
            label_abs = label_abs.cuda().data.cpu().numpy()
            sem_ns = emb_novel[label_abs[:k]]
            label_ns = label_shot
            sem_n1 = emb_novel[lb_abs]
            label_n1 = label_shot[:self.args.way]
            feat_nq = data_query

            if torch.cuda.is_available():
                feat_b = torch.tensor(feat_b).type(torch.cuda.FloatTensor)
                label_b = torch.tensor(label_b).type(torch.cuda.LongTensor)
                sem_b = torch.tensor(sem_b).type(torch.cuda.FloatTensor)
                sem_b1 = torch.tensor(sem_b1).type(torch.cuda.FloatTensor)
                sem_ns = torch.tensor(sem_ns).type(torch.cuda.FloatTensor)
                sem_n1 = torch.tensor(sem_n1).type(torch.cuda.FloatTensor)
            else:
                feat_b = torch.tensor(feat_b).type(torch.FloatTensor)
                label_b = torch.tensor(label_b).type(torch.LongTensor)
                sem_b = torch.tensor(sem_b).type(torch.FloatTensor)
                sem_b1 = torch.tensor(sem_b1).type(torch.FloatTensor)
                sem_ns = torch.tensor(sem_ns).type(torch.FloatTensor)
                sem_n1 = torch.tensor(sem_n1).type(torch.FloatTensor)

            logit_q, emb_n_trans_from_base, feat_n_trans_from_base \
                = self.model((feat_b, sem_b, sem_b1, label_b, feat_ns, sem_ns, label_ns, sem_n1, label_n1, feat_nq))

            if self.args.classifiermethod == 'nonparam':
                acc = np.mean(logit_q == label1)
            else:
                acc = count_acc(logit_q, label)
            ave_acc.add(acc)
            test_acc_record[i - 1] = acc
            if i % 100 == 0:
                print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

        m, pm = compute_confidence_interval(test_acc_record)
        print('Test Acc {:.4f} + {:.4f}'.format(m, pm))
        print('Running Time: {} '.format(timer.measure()))