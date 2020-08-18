import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

structures = pd.read_csv('./data/structures.csv')
test = pd.read_csv('./data/test.csv')

def make_distance_matrix(structures_df, molecule_name):
    mask = structures_df['molecule_name'] == molecule_name
    molecule_structure = structures_df[mask]
    n_atoms = len(molecule_structure)
    
    #get x,y,z coordinates and stack them
    x_coords = molecule_structure.x.values
    y_coords = molecule_structure.y.values
    z_coords = molecule_structure.z.values
    coords = np.stack([x_coords, y_coords, z_coords], axis=1)
    dist_mat = squareform(pdist(coords))
    
    #now we also want to get a matrix with the type of bonds associated with a distance
    pairs = {'CC': 1, 'CF': 2, 'CH': 3, 'CN': 4, 'CO': 5, 'FF': 6, 'FH': 7, 'FN': 8, 
             'FO': 9, 'HH': 10, 'HN': 11, 'HO': 12, 'NN': 13, 'NO': 14, 'OO': 15}
    
    #get atom types for all atoms
    atoms = molecule_structure.atom.values
    
    #create an array of zeros for hold values
    atom_mat = np.zeros_like(dist_mat, dtype=np.uint8)
    for i, a1 in enumerate(atoms):
        for j, a2 in enumerate(atoms):
            pair = ''.join(sorted(a1 + a2))
            value = pairs[pair]
            atom_mat[i][j] = value

    #now compute the distance matrix and convert it to squareform
    return dist_mat, atom_mat, n_atoms

def make_constant_matrix(structures_df, test_df, molecule_name):
    dist_mat, atom_mat, n_atoms = make_distance_matrix(structures_df, molecule_name)
    
    #now constuct a square zeros matrix with size n_atoms x n_atoms
    #constant_mat = np.zeros((n_atoms, n_atoms),dtype=np.float32)
    
    #now get the atom_index_1, atom_index_2, and scalar_coupling_constants
    mask = test_df['molecule_name'] == molecule_name
    molecule_df = test_df[mask]
    index1 = molecule_df.atom_index_0
    index2 = molecule_df.atom_index_1
    #coupling_constant = molecule_df.scalar_coupling_constant
    
    #for i1, i2, const in zip(index1, index2, coupling_constant):
    #    constant_mat[i1][i2] = const
    #    constant_mat[i2][i1] = const
        
    #now also get the types of coupling and store it as a matrix
    type_dict = {'1JHC': 2, '1JHN': 1, '2JHC': 3, '2JHH': 5, '2JHN': 7, '3JHC': 4,'3JHH': 8, '3JHN': 6}
    type_mat = np.zeros((n_atoms, n_atoms), dtype=np.uint8)
    types = molecule_df.type.values
    
    for i1, i2, t in zip(index1, index2, types):
        type_mat[i1][i2] = type_dict[t]
        type_mat[i2][i1] = type_dict[t]
        
    #stack dist_mat, atom_mat, and type_mat into a single array
    input_mat = np.stack([dist_mat, atom_mat, type_mat], axis=0)
        
    return input_mat

#lets generate our distance maps and save them as numpy arrays.
#NOTE: this will take 5 hours to complete!
dist_path = './data/maps/test/dist/'
target_path = './data/maps/test/target/'
molecule_names = np.unique(test.molecule_name)

for mn in tqdm(molecule_names):
    input_mat = make_constant_matrix(structures, test, mn)
    np.save(dist_path + mn + '.npy', input_mat)
    #np.save(target_path + mn + '.npy', constant_mat)