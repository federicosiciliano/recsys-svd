import numpy as np
from sklearn.impute import SimpleImputer
from copy import deepcopy
from sklearn.metrics import jaccard_score
from scipy.sparse.linalg import eigs, svds

import json
from kneed import KneeLocator
import os
import torch
import easy_exp
import time

# Create utility matrix from the train Dataset
def create_utility_matrix_from_dataset(dataset, num_users, num_items):
    utility_matrix_from_dataset = np.zeros((num_users+1, num_items+1)) 
    for x in dataset:
        user = x['uid']
        items = x['sid']
        ratings = x['rating']
        
        for i,item in enumerate(items):
            utility_matrix_from_dataset[user,item] = ratings[i]
        
    return utility_matrix_from_dataset[1:,1:] #FS: removed padding and fake user0: can generate errors?

def compute_and_save_decomposition(utility_matrix, method, **kwargs):
    startime = time.time()
    if method == 'svd':
        out_dict = compute_svd_decomposition(utility_matrix=utility_matrix, **kwargs)
        knee_var = 's'
    elif method == 'leporid':
        out_dict = compute_leporid_decomposition(utility_matrix=utility_matrix, **kwargs)
        knee_var = 'eigenvalues'
    else:
        raise ValueError(f"Invalid decomposition method {method}")
    endtime = time.time()
    
    out_dict['knee'] = compute_knee(out_dict[knee_var])
    out_dict['time'] = endtime - startime
    save_init_results(kwargs["exp_name"], kwargs["exp_id"], kwargs["save_folder"], **out_dict)


def compute_svd_decomposition(utility_matrix, svd_params, mean_imputation_strategy=None, **unused_kwargs):
    # Warn about unused kwargs
    if unused_kwargs: print(f"SVD Warning: unused kwargs {unused_kwargs.keys()}")
    centered_utility_matrix = mean_imputation(utility_matrix,mean_imputation_strategy)

    u, s, v_t = svds(centered_utility_matrix,**svd_params) #np.linalg.svd(centered_utility_matrix, **svd_params)

    v = v_t.transpose()

    return {'u':u, 's':s, 'v':v}

# Fill the missing values with the average rating of either the corresponding user or item
def mean_imputation(utility_matrix, avg_of):
    imputer = SimpleImputer(missing_values=0, strategy='mean')  #ignore zeros in the mean computation treating them as missing values
    if avg_of == 'item':
        imputed_utility_matrix = imputer.fit_transform(utility_matrix)  
    elif avg_of == 'user':
        imputed_utility_matrix = imputer.fit_transform(utility_matrix.T).T 
    elif avg_of == 'both':
        non_zero_elements = utility_matrix[utility_matrix != 0]  
        global_mean = np.mean(non_zero_elements)
        imputed_utility_matrix = np.where(utility_matrix == 0, global_mean, utility_matrix)
    else:
        return utility_matrix
    return imputed_utility_matrix

def compute_knee(s):
    direction_type = check_increasing_or_decreasing(s)
    curve_type = check_concavity_or_convexity(s, if_neither=["neither","convex"][direction_type=="decreasing"])
    
    if direction_type=="neither" or curve_type == "neither":
        return None
    
    kneedle = KneeLocator(range(1,len(s)+1), s, S=1.0, curve=curve_type, direction=direction_type)
    #print(kneedle.knee, kneedle.elbow)
    knee = int(kneedle.knee)
    return knee


def compute_leporid_decomposition(utility_matrix, k, alpha, num_items, **unused_kwargs):
    # warn about unused kwargs
    if unused_kwargs: print(f"Leporid Warning: unused kwargs {unused_kwargs.keys()}")

    # dalla matrice delle utilità ricavo una matrice che ha 1 se c'è stata un'interazione, 0 altrimenti
    interaction_matrix = utility_matrix > 0
    
    # Calcola jaccard similarity per ogni coppia di items
    jaccard_similarity = np.ones((num_items,num_items))
    for i in range(num_items-1):
        a = interaction_matrix[:,i+1:].T
        b = interaction_matrix[None,:,i]
        inters = np.logical_and(a,b).sum(-1)
        union = np.logical_or(a,b).sum(-1)
        jacs = np.divide(inters, union, out=np.zeros_like(union, dtype=float), where=union != 0)
        jaccard_similarity[i+1:,i] = jacs
        jaccard_similarity[i,i+1:] = jacs

    # Data la matrice delle interazioni, calcola per ogni item i k nearest neighbors

    knn_indices = []
    W = np.zeros_like(jaccard_similarity) # inizializza matrice W (delle adiacenze con i k nearest neighbors)

    for i in range(num_items):
        # trova gli indici degli elementi ordinati per similarità decrescente, escludendo l'elemento stesso
        sorted_indices = np.argsort(-jaccard_similarity[i])
        nearest_neighbors = sorted_indices[sorted_indices != i][:k]
        knn_indices.append(nearest_neighbors)
        W[i, nearest_neighbors] = jaccard_similarity[i, nearest_neighbors]

    # matrice diagonale D (degree matrix)
    D = np.diag(W.sum(axis=1))
    d_max = np.max(np.diag(W.sum(axis=1)))

    # matrice laplaciana L
    L = D - W

    # matrice delle identità
    I = np.eye(num_items)  

    # matrice laplaciana regolarizzata
    L_reg = (1 - alpha) * L + alpha * d_max * I

    # eigendecomposition
    eigenvalues, eigenvectors = eigs(L_reg, k=64) #np.linalg.eig(L_reg)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    #order by smallest eigenvalues
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    
    return {'eigenvalues':eigenvalues, 'eigenvectors':eigenvectors}

def get_init_file(exp_name, exp_id, save_folder="../data/processed"):
    init_file = os.path.join(save_folder,exp_name,exp_id+".npz")
    return init_file

def save_init_results(exp_name, exp_id, save_folder="../data/processed", **kwargs):
    init_file = get_init_file(exp_name, exp_id, save_folder)
    init_folder = os.path.join(*init_file.split("/")[:-1])
    if not os.path.exists(init_folder):
        os.makedirs(init_folder)
    np.savez(init_file, **kwargs)


def load_init_results(exp_name, exp_id, save_folder="../data/processed"):
    init_file = get_init_file(exp_name, exp_id, save_folder, )
    return np.load(init_file, allow_pickle=True)

def get_init_cfg(cfg):
    init_cfg = deepcopy(cfg)
    init_cfg["__exp__"]["name"] = "init"
    init_cfg.pop("model")
    init_cfg["init"].pop("training",None)
    init_cfg["data_params"].pop("collator_params")
    for key in deepcopy(init_cfg["__exp__"]["__nosave__"]):
        if init_cfg[key] == {}:
            init_cfg["__exp__"]["__nosave__"].pop(key)
    return init_cfg

# Initialize the item embedding matrix using V (or S*V) generated from SVD on the utility matrix
def create_embedding_matrix(method, emb_size, **kwargs):
    if method == 'svd':
        embedding_matrix = create_embedding_matrix_from_svd(emb_size, **kwargs)
    elif method == 'leporid':
        embedding_matrix = create_embedding_matrix_from_leporid(emb_size, **kwargs)
    else:
        raise ValueError(f"Invalid decomposition method {method}")

    # Add padding for the zero vector (deleted: assume the padding has already been initialized)
    #zero_vector = np.zeros((1, emb_size))
    #embedding_matrix = np.concatenate((zero_vector, embedding_matrix), axis=0)

    return embedding_matrix

def create_embedding_matrix_from_svd(emb_size, s, v, use_diag=False, **unused_kwargs):
    # warn about unused kwargs
    if unused_kwargs: print(f"SVD Warning: unused kwargs {unused_kwargs.keys()}")

    if use_diag:
        embedding_matrix = np.matmul(np.diag(s),np.matrix(v))[:,:emb_size]
    else:
        embedding_matrix = np.matrix(v[:, :emb_size])

    return embedding_matrix


def create_embedding_matrix_from_leporid(emb_size, eigenvectors, **unused_kwargs):
    # warn about unused kwargs
    if unused_kwargs: print(f"Leporid Warning: unused kwargs {unused_kwargs.keys()}")

    embedding_matrix = np.matrix(eigenvectors[:, :emb_size])

    return embedding_matrix


def set_init_embeddings(main_module, embedding_matrix, freeze_emb=False, cutoff=None, knee=None, invert_order=False, **unused_kwargs):
    # warn about unused kwargs
    if unused_kwargs: print(f"Embedding Warning: unused kwargs {unused_kwargs.keys()}")

    #Set the item embedding layer with init right matrix (if freeze_emb=True the matrix weights will remain fixed)
    new_emb_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
    if invert_order:
        new_emb_matrix = torch.flip(new_emb_matrix, dims=[1])
        
    #initialize the item embedding matrix with the new embedding matrix 
    if cutoff:
        print(cutoff,knee)
        if isinstance(cutoff, bool) and cutoff:
            cutoff = knee
        elif cutoff < new_emb_matrix.shape[1]:
            cutoff = new_emb_matrix.shape[1]
        print("CUTOFF",cutoff)
        main_module.item_emb.weight.data[1:new_emb_matrix.shape[0]+1,:cutoff] = new_emb_matrix[:,:cutoff] #1: assumes padding
    else:
        print("NO CUTOFF")
        main_module.item_emb.weight.data[1:new_emb_matrix.shape[0]+1,:] = new_emb_matrix[:,:main_module.item_emb.weight.data.shape[1]] #1: assumes padding

    if freeze_emb:
        for param in main_module.item_emb.parameters():
            param.requires_grad = False
     
def check_increasing_or_decreasing(arr, if_neither="neither"):
    # Compute the first-order finite differences
    first_derivative = np.diff(arr)
    
    # Check the sign of the first derivative
    if np.all(first_derivative >= 0):
        return "increasing"
    elif np.all(first_derivative <= 0):
        return "decreasing"
    else:
        return if_neither

def check_concavity_or_convexity(arr, if_neither="neither"):
    # Compute the second-order finite differences
    second_derivative = np.diff(arr, n=2)
    
    # Check the sign of the second derivative
    if np.all(second_derivative >= 0):
        return "convex"
    elif np.all(second_derivative <= 0):
        return "concave"
    else:
        return if_neither
    

def prepare_embeddings_based_on_init(cfg, main_module, save_folder="../data/processed"):
    if cfg.get("init",False):
        init_cfg = get_init_cfg(cfg)
        init_exp_found, init_experiment_id = easy_exp.exp.get_set_experiment_id(init_cfg)
        if not init_exp_found: raise Exception("Init experiment not found")
        init_results = load_init_results(init_cfg["__exp__"]["name"], init_experiment_id, save_folder)
        
        init_params = deepcopy(cfg["init"])
        init_params.update(init_params.pop("training",{}))
        
        embedding_matrix = create_embedding_matrix(emb_size=cfg["model"]["rec_model"]["emb_size"], **init_params, **init_results)

        set_init_embeddings(main_module, embedding_matrix, **init_params, **init_results)