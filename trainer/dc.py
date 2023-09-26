""" distribution calibration method """
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler
from models.SIFT import Learner
from utils.misc import Averager, Timer, count_acc, compute_confidence_interval
from dataloader.dataset_loader import DatasetLoader as Dataset
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


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


def base_statistics(feat, label):
    ''''Base class statistics'''
    base_means = []
    base_cov = []
    LBs = np.unique(label)
    for lb in LBs:
        id=np.where(label==lb)[0]
        feature=feat[id]
        mean = np.mean(feature, axis=0)
        cov = np.cov(feature.T)
        base_means.append(mean)
        base_cov.append(cov)
    return base_means, base_cov

def distribution_calibration(support, base_means, base_cov, k, alpha=0.21):
    dist = []
    for i in range(len(base_means)):
        dist.append(np.linalg.norm(support-base_means[i]))
    index = np.argpartition(dist, k)[:k]
    mean = np.concatenate([np.array(base_means)[index], support[np.newaxis, :]])
    calibrated_mean = np.mean(mean, axis=0)
    calibrated_cov = np.mean(np.array(base_cov)[index], axis=0)+alpha

    return calibrated_mean, calibrated_cov

class DC(object):
    def __init__(self, args):
        self.args = args
        self.model = Learner(self.args, mode='dc')
        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()

    def eval(self):
        # Load test set
        test_set = Dataset('test', self.args)
        tasknum = 600
        sampler = CategoriesSampler(test_set.label, tasknum, self.args.way, self.args.shot + self.args.val_query)
        loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
        # Set test accuracy recorder
        test_acc_record = np.zeros((tasknum,))
        # Set accuracy averager
        ave_acc = Averager()
        ave_acc0 = Averager()
        '''------------------ Generate labels --------------------'''
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

        '''--------------------- process data --------------------'''
        path = self.args.dataset_dir
        traindata = np.load(path+'/feat-train.npz')
        featbase, labelbase = traindata['features'], traindata['targets']
        train_class = len(np.unique(labelbase))
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
            data_query1 = data_query
            data_query1 = data_query1.cuda().data.cpu().numpy()
            # -------------------------------------------------
            feat_proto = torch.zeros(self.args.way, data_shot.size(1)).type(data_shot.type())
            for lb in torch.unique(label_shot):
                ds = torch.where(label_shot == lb)[0]
                feat_proto[lb] = torch.mean(data_shot[ds], dim=0)
            logit0 = euclidean_metric(data_query, feat_proto)

            '''-------------------- generate data -------------------'''
            support_data = data_shot.cuda().data.cpu().numpy()
            support_label = label_shot.cuda().data.cpu().numpy()
            base_means, base_cov = base_statistics(feat_base, label_base)
            n_lsamples = self.args.way * self.args.shot

            beta = 0.5
            support_data = np.power(support_data, beta)
            data_query1 = np.power(data_query1, beta)
            sampled_data = []
            sampled_label = []
            for j in range(n_lsamples):
                mean, cov = distribution_calibration(support_data[j], base_means, base_cov, k=2)
                sampled_data.append(np.random.multivariate_normal(mean=mean, cov=cov, size=self.args.num_aug))
                sampled_label.extend([support_label[j]] * self.args.num_aug)

            sampled_data = np.concatenate(sampled_data, axis=0)
            sampled_label = np.array(sampled_label)

            X_aug = np.concatenate([support_data, sampled_data])
            Y_aug = np.concatenate([support_label, sampled_label])

            '''------------------ train classifier -------------------'''
            if self.args.classifiermethod == 'nonparam':
                if self.args.cls == 'lr':
                    classifier = LogisticRegression(max_iter=1000).fit(X=X_aug, y=Y_aug)
                elif self.args.cls == 'svm':
                    classifier = SVC(C=10, gamma='auto', kernel='linear', probability=True).fit(X=X_aug, y=Y_aug)
                elif self.args.cls == 'knn':
                    classifier = KNeighborsClassifier(n_neighbors=1).fit(X=X_aug, y=Y_aug)
                predicts = classifier.predict(data_query1)
                acc = np.mean(predicts == label1)
            else:
                if torch.cuda.is_available():
                    X_aug1 = torch.tensor(X_aug).type(torch.cuda.FloatTensor)
                    Y_aug1 = torch.tensor(Y_aug).type(torch.cuda.LongTensor)
                else:
                    X_aug1 = torch.tensor(X_aug).type(torch.FloatTensor)
                    Y_aug1 = torch.tensor(Y_aug).type(torch.LongTensor)
                logit_q = self.model((X_aug1, Y_aug1, data_query))
                acc = count_acc(logit_q, label)
            acc0 = count_acc(logit0, label)
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
