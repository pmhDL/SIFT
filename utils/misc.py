""" Additional utility functions """
import matplotlib
matplotlib.use('pdf')
import os
import time
import pprint
import torch
import numpy as np
import torch.nn.functional as F


def ensure_path(path):
    """The function to make log path.
    Args:
      path: the generated saving path.
    """
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

class Averager():
    """The class to calculate the average."""
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def count_acc(logits, label):
    """The function to calculate the .
    Args:
      logits: input logits.
      label: ground truth labels.
    Return:
      The output accuracy.
    """
    pred = F.softmax(logits, dim=1).argmax(dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    return (pred == label).type(torch.FloatTensor).mean().item()

class Timer():
    """The class for timer."""
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def compute_confidence_interval(data):
    """The function to calculate the .
    Args:
      data: input records
      label: ground truth labels.
    Return:
      m: mean value
      pm: confidence interval.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, v2.T)
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res


def np_proto(feat, label, way):
    feat_proto = np.zeros((way, feat.shape[1]))
    for lb in np.unique(label):
        ds = np.where(label == lb)[0]
        feat_ = feat[ds]
        feat_proto[lb] = np.mean(feat_, axis=0)
    return feat_proto


def tc_proto(feat, label, way):
    feat_proto = torch.zeros(way, feat.size(1))
    for lb in torch.unique(label):
        ds = torch.where(label == lb)[0]
        feat_ = feat[ds]
        feat_proto[lb] = torch.mean(feat_, dim=0)
    if torch.cuda.is_available():
        feat_proto = feat_proto.type(feat.type())
    return feat_proto


def updateproto(Xs, ys, cls_center, way):
    """assign labels for the clusters"""
    proto = np_proto(Xs, ys, way)
    dist = ((proto[:, np.newaxis, :]-cls_center[np.newaxis, :, :])**2).sum(2)
    id = dist.argmin(1)
    feat_proto = np.zeros((way, Xs.shape[1]))
    for i in range(way):
        feat_proto[i] = (cls_center[id[i]] + proto[i])/2
    return feat_proto


def compactness_loss(gen_feat, gen_label, proto, supp_label):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    loss = 0
    for lb in torch.unique(supp_label):
        id = torch.where(gen_label == lb)[0]
        loss = loss + loss_fn(gen_feat[id], proto[lb])
    return loss


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits