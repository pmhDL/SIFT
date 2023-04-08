""" Main function for this repo. """
import argparse
import torch
from utils.misc import pprint
from utils.gpu_tools import set_gpu
from trainer.st import ST
from trainer.dc import DC
from trainer.lr import LR

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic parameters
    parser.add_argument('--model_type', type=str, default='res12', choices=['res12', 'wrn28']) # The network architecture
    parser.add_argument('--dataset', type=str, default='mini', choices=['mini', 'tiered', 'cub', 'cifar_fs']) # Dataset
    parser.add_argument('--phase', type=str, default='st_te', choices=['st_te', 'lr_te', 'dc_te']) # Phase
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', default='1') # GPU id
    parser.add_argument('--dataset_dir', type=str, default='./data/mini/res12') # Dataset folder
    # Parameters for meta-train phase
    parser.add_argument('--shot', type=int, default=1) # Shot number, how many samples for one class in a task
    parser.add_argument('--way', type=int, default=5) # Way number, how many classes in a task
    parser.add_argument('--train_query', type=int, default=15) # The number of training samples for each class in a task
    parser.add_argument('--val_query', type=int, default=15) # The number of test samples for each class in a task
    parser.add_argument('--num_aug', type=int, default=1) #number of augmented samples per class
    parser.add_argument('--classifiermethod', type=str, default='gradient', choices=['metric', 'nonparam', 'gradient'])
    parser.add_argument('--cls', type=str, default='lr', choices=['lr', 'svm', 'knn'])
    parser.add_argument('--selectm', type=str, default='randomselect', choices=['nn2suppcenter', 'nn2basecenter', 'randomselect'])
    parser.add_argument('--setting', type=str, default='in', choices=['tran', 'in'])
    parser.add_argument('--gradlr', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--Ablation', type=str, default='no', choices=['all', 'enc_recon', 'dec_recon', 'cpt', 'no'])
    # Set and print the parameters
    args = parser.parse_args()
    pprint(vars(args))
    # Set the GPU id
    set_gpu(args.gpu)
    # Set manual seed for PyTorch
    if args.seed == 0:
        print('Using random seed.')
        torch.backends.cudnn.benchmark = True
    else:
        print('Using manual seed:', args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Start trainer for pre-train, meta-train or meta-eval
    if args.phase == 'st_te':
        trainer = ST(args)
        trainer.eval()
    elif args.phase == 'lr_te':
        trainer = LR(args)
        trainer.eval()
    elif args.phase == 'dc_te':
        trainer = DC(args)
        trainer.eval()
    else:
        raise ValueError('Please set correct phase.')
