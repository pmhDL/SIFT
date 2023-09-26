'''using visual feature as transfer clue'''
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler
from models.SIFT import Learner
from utils.misc import Averager, Timer, count_acc, compute_confidence_interval, get_cos_similar_matrix, get_cos_similar_matrix_loop
from dataloader.dataset_loader import DatasetLoader as Dataset
import pulp as lp
import time

def euclidean_metric(query, proto):
    '''
    :param a: query
    :param b: proto
    :return: (num_sample, way)
    '''
    n = query.shape[0]  # num_samples
    m = proto.shape[0]  # way
    query = query.unsqueeze(1).expand(n, m, -1)
    proto = proto.unsqueeze(0).expand(n, m, -1)
    logits = -((query - proto) ** 2).sum(dim=2)
    return logits


def route_plan_J(Dij):
    NN, BB = Dij.shape
    model = lp.LpProblem(name='plan_0_1', sense=lp.LpMaximize)
    x = [[lp.LpVariable("x_{},{}".format(i, j), cat="Binary") for j in range(BB)] for i in range(NN)]
    # objective
    objective = 0
    for i in range(NN):
        for j in range(BB):
            objective = objective + Dij[i, j] * x[i][j]
    model += objective
    # constraints
    for i in range(NN):
        in_degree = 0
        for j in range(BB):
            in_degree = in_degree + x[i][j]
        model += in_degree == 1
    for j in range(BB):
        out_degree = 0
        for i in range(NN):
            out_degree = out_degree + x[i][j]
        model += out_degree <= 1
    model.solve(lp.apis.PULP_CBC_CMD(msg=False))

    W = np.zeros((NN, BB))
    for v in model.variables():
        idex = [int(s) for s in v.name.split('_')[1].split(',')]
        W[idex[0], idex[1]] = v.varValue
    return W


class NS(object):
    def __init__(self, args):
        self.args = args
        self.model = Learner(self.args, mode='ns')
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
        return feat_b, label_b

    def eval(self):
        # Load test set
        tasknum = 600
        test_set = Dataset('test', self.args)
        sampler = CategoriesSampler(test_set.label, tasknum, self.args.way, self.args.shot + self.args.val_query)
        loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
        # Set test accuracy recorder
        test_acc_record = np.zeros((tasknum,))
        # Set accuracy averager
        ave_acc = Averager()
        ave_acc0 = Averager()

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
        train_classes = np.unique(labelbase)
        base_class_mean = []
        for lb in train_classes:
            index1 = np.where(labelbase == lb)[0]
            base_class_mean.append(featbase[index1].mean(0))
        base_class_mean = np.array(base_class_mean)
        #------------------------------------------
        feat_base = featbase
        label_base = labelbase

        timer = Timer()
        start_time = time.time()
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
            # # ----------------------------------------------------
            feat_proto = torch.zeros(self.args.way, data_shot.size(1)).type(data_shot.type())
            for lb in torch.unique(label_shot):
                ds = torch.where(label_shot == lb)[0]
                feat_proto[lb] = torch.mean(data_shot[ds], dim=0)
            logit0 = euclidean_metric(data_query, feat_proto)
            feat_proto_np = feat_proto.cuda().data.cpu().numpy()

            # ---------------compute similarity and select base classes-----------------
            vis_novel_base = get_cos_similar_matrix(feat_proto_np, base_class_mean)
            maxid = np.where(route_plan_J(vis_novel_base) == 1)[1]

            '''----------------- generate samples ------------------'''
            base_lb_abs = maxid
            feat_b, label_b = self.sample_base_data1(feat_base, label_base, base_lb_abs, feat_proto_np, self.args.num_aug)
            feat_ns = data_shot
            label_ns = label_shot
            feat_nq = data_query

            if torch.cuda.is_available():
                feat_b = torch.tensor(feat_b).type(torch.cuda.FloatTensor)
                label_b = torch.tensor(label_b).type(torch.cuda.LongTensor)
            else:
                feat_b = torch.tensor(feat_b).type(torch.FloatTensor)
                label_b = torch.tensor(label_b).type(torch.LongTensor)

            logit_q = self.model((feat_b, label_b, feat_ns, label_ns, feat_nq))
            acc0 = count_acc(logit0, label)
            if self.args.classifiermethod == 'nonparam':
                acc = np.mean(logit_q == label1)
            else:
                acc = count_acc(logit_q, label)
            ave_acc.add(acc)
            ave_acc0.add(acc0)
            test_acc_record[i - 1] = acc
            if i % 100 == 0:
                print('batch {}: {:.2f} {:.2f} '.format(i, ave_acc0.item() * 100, ave_acc.item() * 100))
        end_time = time.time()
        time_mean = (end_time - start_time) / tasknum
        m, pm = compute_confidence_interval(test_acc_record)
        print('Test Acc {:.4f} + {:.4f}'.format(m, pm))
        print('per task runing time: ', time_mean)
        print('Running Time: {} '.format(timer.measure()))
