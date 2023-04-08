import os.path as osp
from torch.utils.data import Dataset
import numpy as np

class DatasetLoader(Dataset):
    """The class to load the dataset"""
    def __init__(self, setname, args):
        THE_PATH = osp.join(args.dataset_dir, 'feat-' + setname + '.npz')
        data0 = np.load(THE_PATH)
        data = data0['features']
        label = data0['targets']

        self.data = data
        self.label = label
        self.num_class = len(np.unique(label))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image = self.data[i]
        label = self.label[i]
        return image, label