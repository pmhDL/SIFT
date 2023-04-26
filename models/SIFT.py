import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import pulp as lp
from utils.misc import euclidean_metric, compactness_loss, tc_proto, np_proto


def route_plan(Dij):
    K = Dij.shape[0]
    model = lp.LpProblem(name='plan_0_1', sense=lp.LpMinimize)
    x = [[lp.LpVariable("x{}{}".format(i, j), cat="Binary") for j in range(K)] for i in range(K)]
    # objective
    objective = 0
    for i in range(K):
        for j in range(K):
            objective = objective + Dij[i, j] * x[i][j]
    model += objective
    # constraints
    for i in range(K):
        in_degree = 0
        for j in range(K):
            in_degree = in_degree + x[i][j]
        model += in_degree == 1

    for i in range(K):
        out_degree = 0
        for j in range(K):
            out_degree = out_degree + x[j][i]
        model += out_degree == 1

    model.solve(lp.apis.PULP_CBC_CMD(msg=False))

    W = np.zeros((K, K))
    i = 0
    j = 0
    for v in model.variables():
        W[i, j] = v.varValue
        j = j + 1
        if j % K == 0:
            i = i + 1
            j = 0
    return W


def updateproto_(Xs, ys, cls_center, way):
    """assign labels for the clusters based on assignment method"""
    proto = np_proto(Xs, ys, way)
    dist = ((proto[:, np.newaxis, :]-cls_center[np.newaxis, :, :])**2).sum(2)
    W = route_plan(dist)
    _, id = np.where(W > 0)
    feat_proto = np.zeros((way, Xs.shape[1]))
    for i in range(way):
        feat_proto[i] = (proto[i] + cls_center[id[i]])/2
    return feat_proto


class Classifier(nn.Module):

    def __init__(self, way, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.way = way
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.way, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(self.way))
        self.vars.append(self.fc1_b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        net = F.linear(input_x, fc1_w, fc1_b)
        return net

    def parameters(self):
        return self.vars


class FClayer(nn.Module):

    def __init__(self, z_out, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.z_out = z_out
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.z_out, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        # self.fc1_b = nn.Parameter(torch.zeros(self.z_out))
        # self.vars.append(self.fc1_b)
    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        #fc1_b = the_vars[1]
        #net = F.linear(input_x, fc1_w, fc1_b)
        net = F.linear(input_x, fc1_w)
        return net

    def parameters(self):
        return self.vars


class Learner(nn.Module):

    def __init__(self, args, mode='st'):
        super().__init__()
        self.args = args
        self.mode = mode
        z_dim = 640
        if self.args.dataset == 'cub':
            z_sem = 312
        else:
            z_sem = 300

        if mode == 'st':
            self.fc_en = FClayer(z_sem, z_dim)
            self.activefun1 = nn.ReLU()
            self.trans = nn.Linear(z_sem, z_sem) # self.trans = FClayer(z_sem, z_sem)
            self.activefun2 = nn.Sigmoid() # nn.ReLU()
            self.fc_de = FClayer(z_dim, z_sem)
            self.classifyer = Classifier(self.args.way, z_dim)
        elif mode == 'dc':
            self.classifyer = Classifier(self.args.way, z_dim)

    def forward(self, inp):
        """The function to forward the model."""
        if self.mode == 'st':
            feat_b, sem_b, sem_b1, label_b, feat_ns, sem_ns, label_ns, sem_n1, label_n1, feat_nq = inp
            return self.st_forward(feat_b, sem_b, sem_b1, label_b, feat_ns, sem_ns, label_ns, sem_n1, label_n1, feat_nq)
        elif self.mode == 'dc':
            feat_s, label_s, feat_q = inp
            return self.dc_forward(feat_s, label_s, feat_q)
        else:
            raise ValueError('Please set the correct mode.')


    def dc_forward(self, feat_s, label_s, feat_q):

        if self.args.classifiermethod == 'gradient':
            logits = self.classifyer(feat_s)
            loss = F.cross_entropy(logits, label_s)
            grad = torch.autograd.grad(loss, self.classifyer.parameters())
            fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, self.classifyer.parameters())))

            for _ in range(1, 100):
                logits = self.classifyer(feat_s, fast_weights)
                loss = F.cross_entropy(logits, label_s)
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, fast_weights)))
            logits_q = self.classifyer(feat_q, fast_weights)

        elif self.args.classifiermethod == 'metric':
            protos = tc_proto(feat_s, label_s, self.args.way)
            logits_q = euclidean_metric(feat_q, protos)

        return logits_q


    def st_forward(self, feat_b, sem_b, sem_b1, label_b, feat_ns, sem_ns, label_ns, sem_n1, label_n1, feat_nq):
        '''
        :param feat_b: base features (way*N, 640), N: the selected samples per class
        :param sem_b: semantic features of base samples (way*N, 300)
        :param sem_b1: semantic features of base classes (way, 300)
        :param label_b: base labels (way*N, )
        :param feat_ns: support features (way*shot, 300)
        :param sem_ns: semantic features (way*shot, 300)
        :param label_ns: support labels (way*shot, )
        :param sem_n1: ground truth semantic features of each class
        :param label_n1: labels of each class (way, )
        :param feat_nq: query features
        :return: feat_n_1_1 the generated features
        '''
        # transductive
        Xq = feat_nq.cuda().data.cpu().numpy()
        Xs = feat_ns.cuda().data.cpu().numpy()
        ys = label_ns.cuda().data.cpu().numpy()
        if self.args.shot == 1:
            km = KMeans(n_clusters=self.args.way, max_iter=1000, random_state=100)
        else:
            p_np = np_proto(Xs, ys, self.args.way)
            km = KMeans(n_clusters=self.args.way, init=p_np, max_iter=1000, random_state=100)
        #km = KMeans(n_clusters=self.args.way, max_iter=1000, random_state=100)
        yq_fit = km.fit(Xq)
        clus_center = yq_fit.cluster_centers_
        proto1 = updateproto_(Xs, ys, clus_center, self.args.way)
        proto1 = torch.tensor(proto1).type(feat_ns.type())
        proto1 = F.normalize(proto1, dim=1)

        # inductive
        proto2 = tc_proto(feat_ns, label_ns, self.args.way)
        proto2 = F.normalize(proto2, dim=1)

        if self.args.setting == 'tran':
            proto = proto1
        elif self.args.setting == 'in':
            proto = proto2

        loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
        optimizer = torch.optim.Adam([{'params': self.fc_en.parameters(), 'lr': self.args.lr},
                                      {'params': self.trans.parameters(), 'lr': self.args.lr},
                                      {'params': self.fc_de.parameters(), 'lr': self.args.lr}], lr=self.args.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

        for i in range(50):
            lr_scheduler.step()
            self.fc_en.train()
            self.trans.train()
            self.fc_de.train()
            optimizer.zero_grad()

            '''-----------losses of encoder---------'''
            # mapping constraint
            sem_b_1 = self.fc_en(feat_b)   # sem_b_1 = self.activefun2(sem_b_1)
            loss1 = loss_fn(sem_b, sem_b_1)
            # reconstruction constraint
            vars = nn.ParameterList()
            fc1_w = nn.Parameter(self.fc_en.fc1_w.transpose(1, 0))
            vars.append(fc1_w)
            feat_b_1 = self.fc_en(sem_b_1, vars)
            loss2 = loss_fn(feat_b, feat_b_1)

            '''--------------loss of trans-------------'''
            # mapping constraint
            sem_n_1 = self.trans(sem_b1)  # sem_n_1=self.activefun(sem_n_1)
            loss3 = loss_fn(sem_n1, sem_n_1)

            '''--------losses of decoder --------'''
            # mapping constraint
            feat_ns_1 = self.fc_de(sem_ns)
            loss5 = loss_fn(feat_ns, feat_ns_1)
            # reconstruction constraint
            vars = nn.ParameterList()
            fc1_w = nn.Parameter(self.fc_de.fc1_w.transpose(1, 0))
            vars.append(fc1_w)
            sem_ns_1 = self.fc_de(feat_ns_1, vars)
            loss6 = loss_fn(sem_ns, sem_ns_1)

            '''------------ compactness -----------'''
            # transform the samples from base classes to novel classes
            sem_n_1_1 = self.trans(sem_b_1) # sem_n_1_1 = self.activefun2(sem_n_1_1)
            feat_n_1_1 = self.fc_de(sem_n_1_1)
            loss7 = compactness_loss(feat_n_1_1, label_b, proto, label_ns)

            '''------------ablation study-------------'''
            if self.args.Ablation == 'no':
                loss = loss1 + loss2 + loss3 + loss5 + loss6 + loss7
            elif self.args.Ablation == 'enc_recon':
                loss = loss1 + loss3 + loss5 + loss6 + loss7
            elif self.args.Ablation == 'dec_recon':
                loss = loss1 + loss2 + loss3 + loss5 + loss7
            elif self.args.Ablation == 'cpt':
                loss = loss1 + loss2 + loss3 + loss5 + loss6
            elif self.args.Ablation == 'all':
                loss = loss1 + loss3 + loss5

            loss.backward(retain_graph=True)
            optimizer.step()

            '''-------------- augmented novel support ------------'''
            feat = torch.cat((feat_n_1_1, feat_ns), dim=0)
            labels = torch.cat((label_b, label_ns), dim=0)
            feat = F.normalize(feat, dim=1)

            '''----------- train classifier -----------'''
            if self.args.classifiermethod == 'gradient':
                logits = self.classifyer(feat)
                loss = F.cross_entropy(logits, labels)
                grad = torch.autograd.grad(loss, self.classifyer.parameters())
                fast_weights = list(map(lambda p: p[1] - self.args.gradlr * p[0], zip(grad, self.classifyer.parameters())))

                for _ in range(1, 100):
                    logits = self.classifyer(feat, fast_weights)
                    loss = F.cross_entropy(logits, labels)
                    grad = torch.autograd.grad(loss, fast_weights)
                    fast_weights = list(map(lambda p: p[1] - self.args.gradlr * p[0], zip(grad, fast_weights)))
                logits_q = self.classifyer(feat_nq, fast_weights)

            elif self.args.classifiermethod == 'metric': # protoNet
                protos = tc_proto(feat, labels, self.args.way)
                logits_q = euclidean_metric(feat_nq, protos)

            elif self.args.classifiermethod == 'nonparam': # LR, SVM, KNN
                X_aug = feat.cuda().data.cpu().numpy()
                Y_aug = labels.cuda().data.cpu().numpy()
                data_query1 = feat_nq.cuda().data.cpu().numpy()
                if self.args.cls == 'lr':
                    classifier = LogisticRegression(max_iter=1000).fit(X=X_aug, y=Y_aug)
                elif self.args.cls == 'svm':
                    classifier = SVC(C=10, gamma='auto', kernel='linear', probability=True).fit(X=X_aug, y=Y_aug)
                elif self.args.cls == 'knn':
                    classifier = KNeighborsClassifier(n_neighbors=1).fit(X=X_aug, y=Y_aug)
                logits_q = classifier.predict(data_query1)
            else:
                raise ValueError('Please set the correct method.')

        return logits_q, sem_n_1_1, feat_n_1_1
