import torch
import numpy as np
from GenFeat.backbone import Res12, WRN28
from GenFeat.data_loader import DatasetLoader
import os

def extract_feature(data_loader, setname, model, savepath):
    model.eval()
    feat = []
    Lb = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            outputs = model(inputs)
            outputs = outputs.cpu().data.numpy()
            labels = labels.cpu().data.numpy()
            feat.append(outputs)
            Lb.extend(labels)
        feat = np.concatenate(feat, axis=0)
        Lb = np.array(Lb)
    print('feat shape: ', feat.shape)
    print('Lb shape: ', Lb.shape)
    np.savez(savepath + '/feat-'+setname+'.npz', features=feat, targets=Lb)
    return 0

'''------------------------params---------------------------'''
dataname = 'mini'    # mini, tiered, cub, cifar_fs
modeltype = 'res12'  # wrn28 res12
datadir = './data/'+dataname
checkpointpath = './checkpoints/'+dataname+'/'+modeltype+'.pth'
savepath = './data/' + dataname + '/' + modeltype

cuda_device = '0'
'''----------------------construct model---------------------'''
if modeltype == 'res12':
    model = Res12()
elif modeltype == 'wrn28':
    model = WRN28()

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    model = model.cuda()

os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
print('Using gpu:', cuda_device)
'''--------------------load checkpoints----------------------'''
model_dict = model.state_dict()
#print('model_dict: ', model_dict.keys())

checkpoint = torch.load(checkpointpath)
state = checkpoint['params'] # params, state
#print('state: ', state.keys())
#state = {k.replace('encoder.', ''): v for k, v in state.items()}
state = {k: v for k, v in state.items() if k in model_dict}
print('state: ', state.keys())

model_dict.update(state)
model.load_state_dict(model_dict)

'''----------------load data and extract features-------------'''
setname = 'train'
dataset = DatasetLoader(setname, datadir)
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=128,
                                          shuffle=False, num_workers=12, pin_memory=True)

extract_feature(data_loader, setname, model, savepath)


