import torch
import numpy as np
from torch.utils.data import Dataset

class Transform:
    def __init__(self, transforms=None):
        self.transforms = transforms
        
    def __call__(self, matrices, graph, target):
        for transform in self.transforms:
            matrices, graph, target = transform(matrices, graph, target)
        
        return matrices, graph, target
    
class Pad:
    def __init__(self, pad_size=29):
        self.pad_size = pad_size
        
    def calculate_padding(self, matrix_size):
        total_padding = self.pad_size - matrix_size
        
        if total_padding % 2 == 0:
            before = after = total_padding / 2
        else:
            before = (total_padding / 2) + 0.5
            after = total_padding - before

        return int(before), int(after)
        
    def __call__(self, matrices, graph, target):
        before, after = self.calculate_padding(target.shape[0])
        matrices = np.pad(matrices, ((0, 0), (before, after), (before, after)), mode='constant')
        graph = np.pad(graph, (before, after), mode='constant')
        target = np.pad(target, (before, after), mode='constant')
        
        return matrices, graph, target

class MolData(Dataset):
    def __init__(self, fnames, inpath, graph_path, outpath=None, transforms=None, norms=None):
        super(MolData, self).__init__()
        self.fnames = fnames
        self.inpath = inpath
        self.outpath = outpath
        self.graph_path = graph_path
        self.transforms = transforms
        self.norms = norms

    def __len__(self):
        return len(self.fnames)
        
    def __getitem__(self, idx):
        #first load the numpy array data
        fname = self.fnames[idx]
        in_file = self.inpath + '/' + fname
        matrices = np.load(in_file)
        
        graph = np.load(self.graph_path + '/' + fname)
        
        #this is needed
        graph[np.where(graph == 0)] = -1
        
        if self.outpath:
            target_file = self.outpath + '/' + fname
            target = np.load(target_file)
        else:
            target = np.zeros(matrices.shape[1:])
            
        #apply transforms
        if self.transforms is not None:
            matrices, graph, target = self.transforms(matrices, graph, target)
            
        #the first layer is the distance matrix
        #second layer are the atom/bond types
        #third layer are the coupling types
        dist_mat, bond_mat, type_mat = matrices
        
        #now we randomly select a coordinate pair in target that is not zero
        target_idx = np.array(np.where(target != 0)).T
        choice = np.random.choice(range(len(target_idx)))
        target_idx = target_idx[choice]
        
        
        atom0, atom1 = target_idx[0], target_idx[1]
        target = target[atom0, atom1]

        #TODO: for each row get the indices where the distance is less than the defined cutoff distance
        
        #now we need to convert our data to torch tensors
        #dist_mat and target are converted to float
        #dist_mat needs to be normalized
        #bond_mat and type_mat need to be converted to long type
        dist_mat0 = torch.tensor(dist_mat[atom0]).float() - self.norms[0] / self.norms[1]
        dist_mat1 = torch.tensor(dist_mat[atom1]).float() - self.norms[0] / self.norms[1]
        graph0 = torch.tensor(graph[atom0]).long()
        graph1 = torch.tensor(graph[atom1]).long()
        bond0 = torch.tensor(bond_mat[atom0]).long()
        bond1 = torch.tensor(bond_mat[atom1]).long()
        ctype = torch.tensor(type_mat[atom0, atom1]).long()
        
        return {'fname': fname,
                'index0': atom0,
                'index1': atom1,
                'distance0': dist_mat0,
                'distance1': dist_mat1,
                'graph0': graph0,
                'graph1': graph1,
                'bond0': bond0,
                'bond1': bond1,
                'type': ctype,
                'target': target
               }