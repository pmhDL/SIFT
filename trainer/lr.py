""" Train the classfier with support samples """
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler
from models.SIFT import Classifier
from utils.misc import Averager, count_acc, compute_confidence_interval, euclidean_metric, tc_proto, updateproto
from dataloader.dataset_loader import DatasetLoader as Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans


class LR(object):
    def __init__(self, args):
        self.args = args
        z_dim = 640
        self.classifyer = Classifier(self.args.way, z_dim)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.classifyer = self.classifyer.cuda()

    def eval(self):
        # Load test set
        test_set = Dataset('test', self.args)
        sampler = CategoriesSampler(test_set.label, 600, self.args.way, self.args.shot + self.args.val_query)
        loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
        # Set test accuracy recorder
        test_acc_record = np.zeros((600,))
        # Set accuracy averager
        ave_acc = Averager()

        '''---------------Generate labels --------------'''
        # Generate labels
        label = torch.arange(self.args.way).repeat(self.args.val_query)
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
            label_shot = label_shot.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)
            label_shot = label_shot.type(torch.LongTensor)

        for i, batch in enumerate(loader, 1):
            if torch.cuda.is_available():
                data, label_abs = [_.cuda() for _ in batch]
            else:
                data, label_abs = batch[0], batch[1]
            k = self.args.way * self.args.shot
            data_shot, data_query = data[:k], data[k:]

            data_shot = F.normalize(data_shot, dim=1)
            data_query = F.normalize(data_query, dim=1)

            '''--------------- prototypes --------------'''
            if self.args.setting == 'in':
                proto = tc_proto(data_shot, label_shot, self.args.way)
                proto = F.normalize(proto, dim=1)
            elif self.args.setting == 'tran':
                Xs = data_shot.cuda().data.cpu().numpy()
                ys = label_shot.cuda().data.cpu().numpy()
                Xq = data_query.cuda().data.cpu().numpy()
                km = KMeans(n_clusters=self.args.way, max_iter=1000, random_state=100)
                yq_fit = km.fit(Xq)
                clus_center = yq_fit.cluster_centers_
                proto = updateproto(Xs, ys, clus_center, self.args.way)
                proto = torch.tensor(proto).type(data_shot.type())
                proto = F.normalize(proto, dim=1)

            '''---------- train classifier -----------'''
            if self.args.classifiermethod == 'gradient':
                logits = self.classifyer(data_shot)
                loss = F.cross_entropy(logits, label_shot)
                grad = torch.autograd.grad(loss, self.classifyer.parameters())
                fast_weights = list(map(lambda p: p[1] - self.args.gradlr * p[0], zip(grad, self.classifyer.parameters())))
                fast_weights[0].data = proto
                for _ in range(1, 100):
                    logits = self.classifyer(data_shot, fast_weights)
                    loss = F.cross_entropy(logits, label_shot)
                    grad = torch.autograd.grad(loss, fast_weights)
                    fast_weights = list(map(lambda p: p[1] - self.args.gradlr * p[0], zip(grad, fast_weights)))
                logit_q = self.classifyer(data_query, fast_weights)

            elif self.args.classifiermethod == 'metric':
                logit_q = euclidean_metric(data_query, proto)

            elif self.args.classifiermethod == 'nonparam':
                X = data_shot.cuda().data.cpu().numpy()
                Y = label_shot.cuda().data.cpu().numpy()
                X_t = data_query.cuda().data.cpu().numpy()
                Y_t = label.cuda().data.cpu().numpy()
                if self.args.cls == 'lr':
                    classifier = LogisticRegression(max_iter=1000).fit(X=X, y=Y)
                elif self.args.cls == 'svm':
                    classifier = SVC(C=10, gamma='auto', kernel='linear', probability=True).fit(X=X, y=Y)
                elif self.args.cls == 'knn':
                    classifier = KNeighborsClassifier(n_neighbors=1).fit(X=X, y=Y)
                logit_q = classifier.predict(X_t)

            if self.args.classifiermethod == 'nonparam':
                acc = np.mean(logit_q == Y_t)
            else:
                acc = count_acc(logit_q, label)

            ave_acc.add(acc)
            test_acc_record[i - 1] = acc
            if i % 100 == 0:
                print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

        m, pm = compute_confidence_interval(test_acc_record)
        print('Test Acc {:.4f} + {:.4f}'.format(m, pm))
