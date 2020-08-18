import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob

class Transform:
    def __init__(self, transforms=None):
        self.transforms = transforms
        
    def __call__(self, matrices, graph, target):
        for transform in self.transforms:
            matrices, graph, target = transform(matrices, graph, target)
        
        return matrices, graph, target
    

def target_normalize(target, ctype, type_norms):
    for i in range(1, 9):
        if (ctype == i).sum() > 0:
            target[np.where(ctype == i)] = (target[np.where(ctype == i)] - type_norms[i-1][0]) / type_norms[i-1][1]
        
    return target
    
class Normalize:
    def __init__(self, norms=None):
        self.norms = norms
        
    def __call__(self, matrices, graph, target):
        dist_mat = (matrices[0] - self.norms[0]) / self.norms[1]
        matrices[0] = dist_mat
        
        return matrices, graph, target
    
class Cutout:
    def __init__(self, max_cutouts=1, max_cutout_size=8):
        self.max_cutouts = max_cutouts
        self.max_cutout_size = max_cutout_size
        
    def cutout_coordinates(self, matrices):
        matrix_size = matrices.shape[-1]
        xs = np.random.randint(0, matrix_size - 1)
        ys = np.random.randint(0, matrix_size - 1)
        
        shift = np.random.randint(1, self.max_cutout_size)
        xe = xs + shift
        ye = ys + shift
        
        return xs, ys, xe, ye
        
    def __call__(self, matrices, target):
        n_cutouts = np.random.randint(0, self.max_cutouts)
        for _ in range(n_cutouts):
            xs, ys, xe, ye = self.cutout_coordinates(matrices)
            matrices[:, xs:xe, ys:ye] = 0
            target[xs:xe, ys:ye] = 0
        
        return matrices, target
    
def calculate_padding(matrix_size, pad_size):
    total_padding = pad_size - matrix_size

    if total_padding % 2 == 0:
        before = after = total_padding / 2
    else:
        before = (total_padding / 2) + 0.5
        after = total_padding - before

    return int(before), int(after)
    
    
class MolData(Dataset):
    def __init__(self, fnames, inpath, graph_path, node_path, outpath=None, pad_size=29, norms=None, type_norms=None):
        super(MolData, self).__init__()
        self.fnames = np.array(fnames)
        self.inpath = inpath
        self.graph_path = graph_path
        self.node_path = node_path
        self.outpath = outpath
        
        self.pad_size = pad_size
        self.norms = norms
        self.type_norms = type_norms
    
        
    def __len__(self):
        return len(self.fnames)
        
    def __getitem__(self, idx):
        #first load the numpy array data
        fname = self.fnames[idx]
        in_file = self.inpath + '/' + fname
        matrices = np.load(in_file)
        graph = np.load(self.graph_path + '/' + fname)
        nodes = np.load(self.node_path + '/' + fname)
        
        #this is needed
        graph[np.where(graph == 0)] = 1
        
        if self.outpath:
            target_file = self.outpath + '/' + fname
            target = np.load(target_file)
        else:
            target = np.zeros(matrices.shape[1:])
            
        if self.norms is not None:
            matrices[0] = (matrices[0] - self.norms[0]) / self.norms[1]
        
        if self.type_norms is not None:
            target = target_normalize(target, type_mat, self.type_norms)
            
        if self.pad_size is not None:    
            before, after = calculate_padding(target.shape[0], self.pad_size)
            matrices = np.pad(matrices, ((0, 0), (before, after), (before, after)), mode='constant')
            graph = np.pad(graph, (before, after), mode='constant')
            nodes = np.pad(nodes, (before, after), mode='constant')
            target = np.pad(target, (before, after), mode='constant')    

        
        #the first layer is the distance matrix
        #second layer are the atom/bond types
        #third layer are the coupling types
        dist_mat, _, type_mat = matrices
        
        #we only need to take the first row of the bond_mat
        #since the atom types are the same for each row
        
        
        #now we need to convert our data to torch tensors
        #dist_mat and target are converted to float
        #dist_mat needs to be normalized
        #bond_mat and type_mat need to be converted to long type
        dist_mat = torch.tensor(dist_mat).float()
        target = torch.tensor(target).float()
        #bond_mat = torch.tensor(bond_mat).long()
        nodes = torch.tensor(nodes).long()
        type_mat = torch.tensor(type_mat).long()
        graph = torch.tensor(graph).float()
        
        return {'fname': fname,
                'distance': dist_mat,
                'atoms': nodes,
                'type': type_mat,
                'graph': graph,
                'target': target
               }