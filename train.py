import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from glob import glob
#from tqdm import tqdm
from matplotlib import pyplot as plt

from model import *
from data import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

graph_path = './data/maps/paths/'
node_path = './data/maps/nodes/'
trn_fnames = next(os.walk('./data/maps/train/dist/'))[2]
trn_inpath = './data/maps/train/dist/'
trn_outpath = './data/maps/train/target/'

val_fnames = next(os.walk('./data/maps/valid/dist/'))[2]
val_inpath = './data/maps/valid/dist/'
val_outpath = './data/maps/valid/target/'

tst_fnames = next(os.walk('./data/maps/test/dist/'))[2]
tst_inpath = './data/maps/test/dist/'

norms = (3.0482845, 0.27204284)
type_norms = [(47.479884, 10.922172), (94.976153, 18.277237), (-0.270624, 4.523611), 
             (3.688470, 3.070907), (-10.286605, 3.979607), (0.990730, 1.315393),
             (3.124754, 1.315393), (4.771023, 3.704984)]

trn_data = MolData(trn_fnames, trn_inpath, graph_path, node_path, trn_outpath, 29, norms=norms)
val_data = MolData(val_fnames, val_inpath, graph_path, node_path, val_outpath, 29, norms=norms)
tst_data = MolData(tst_fnames, tst_inpath, graph_path, node_path, None, 29)

train = DataLoader(trn_data, batch_size=128, shuffle=True)
valid = DataLoader(val_data, batch_size=128)
#test = DataLoader(tst_data, batch_size=1)

gcn = SemGCN(3, 5, inplanes=128, n_blocks=36, dilations=[1, 2, 3])
#model.load_state_dict(torch.load('./models/augs1.pt'))
loss = SmoothL1Types()
#loss = MSETypes()
#loss = MSE()
optim = torch.optim.Adam

#gcn.load_state_dict(torch.load('./models/test.pt'))

learner = Learner(gcn, loss, optim, train, valid=valid)

save_path = './models/semgcn_1c_22k.pt'
#learner.training(epochs=1, iters=1, lr=1e-2, save_path=save_path)
learner.train_one_cycle(22000, 3e-3, eval_iters=1000, save_path=save_path)
