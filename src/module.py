import numpy as np
from sklearn.impute import SimpleImputer
from copy import deepcopy
from sklearn.metrics import jaccard_score
from scipy.sparse.linalg import eigsh

import json
from kneed import KneeLocator
import os


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
    if method == 'svd':
        compute_svd_decomposition(utility_matrix=utility_matrix, **kwargs)
    elif method == 'leporid':
        compute_leporid_decomposition(utility_matrix=utility_matrix, **kwargs)
    else:
        raise ValueError(f"Invalid decomposition method {method}")


def compute_svd_decomposition(utility_matrix, exp_name, exp_id, mean_imputation_strategy=None, processed_data_folder="../data/processed"):
    centered_utility_matrix = mean_imputation(utility_matrix,mean_imputation_strategy),

    u, s, v_t = np.linalg.svd(centered_utility_matrix)

    v = v_t.transpose()

    knee = compute_knee(s)

    save_init_results(processed_data_folder, exp_name, exp_id, knee = knee, u=u, s=s, v=v)

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
    kneedle = KneeLocator(range(1,len(s)+1), s, S=1.0, curve="convex", direction="decreasing")
    #print(kneedle.knee, kneedle.elbow)
    knee = int(kneedle.knee)
    return knee


def compute_leporid_decomposition(utility_matrix, k, alpha, exp_name, exp_id, num_items, num_users, processed_data_folder="../data/processed"):
    # dalla matrice delle utilità ricavo una matrice che ha 1 se c'è stata un'interazione, 0 altrimenti
    interaction_matrix = (utility_matrix > 0).astype(int)

    # Calcola jaccard similarity per ogni coppia di items
    jaccard_similarity = np.zeros((num_items, num_items))
    for i in range(num_items):
        for j in range(i + 1, num_items):
            similarity = jaccard_score(interaction_matrix[:, i], interaction_matrix[:, j], zero_division=1)
            jaccard_similarity[i, j] = similarity
            jaccard_similarity[j, i] = similarity

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
    eigenvalues, eigenvectors = eigsh(L_reg, k=emb_size, which='SM')

    return eigenvalues, eigenvectors


def get_init_file(processed_data_folder, exp_name, exp_id):
    init_file = os.path.join(processed_data_folder,exp_name,exp_id+".npz")
    return init_file

def save_init_results(processed_data_folder, exp_name, exp_id, **kwargs):
    init_file = get_init_file(processed_data_folder, exp_name, exp_id)
    init_folder = os.path.join(*init_file.split("/")[:-1])
    if not os.path.exists(init_folder):
        os.makedirs(init_folder)
    np.savez(init_file, **kwargs)


#def load_init_results(...):


def load_init_results(processed_data_folder, exp_name, exp_id):
    init_file = get_init_file(processed_data_folder, exp_name, exp_id)
    return np.load(init_file)



def get_init_cfg(cfg):
    init_cfg = deepcopy(cfg)
    init_cfg["__exp__"]["name"] = "init"
    init_cfg.pop("model")
    init_cfg["data_params"].pop("collator_params")
    for key in deepcopy(init_cfg["__exp__"]["__nosave__"]):
        if init_cfg[key] == {}:
            init_cfg["__exp__"]["__nosave__"].pop(key)
    return init_cfg





# REDO...
# Initialize the item embedding matrix using V (or S*V) generated from SVD on the utility matrix
def create_embedding_matrix(utility_matrix, emb_size, use_diag=False):
    U, S, V_t = np.linalg.svd(utility_matrix)

    V = V_t.transpose()
    #embedding_matrix = np.matrix(V_t[:, :emb_size])
    if use_diag:
        embedding_matrix = np.matmul(np.diag(S),np.matrix(V))[:,:emb_size]
    else:
        embedding_matrix = np.matrix(V[:, :emb_size])

    # Add padding for the zero vector #FS: modified to add padding
    zero_vector = np.zeros((1, emb_size))
    embedding_matrix = np.concatenate((zero_vector, embedding_matrix), axis=0)

    return embedding_matrix



    # # scarto il primo autovettore (autovettore associato all'autovalore zero)
    # embedding = np.matrix(eigenvectors[:, :emb_size])



def set_svd_embeddings(model_params, svd_results):
    #Set the item embedding layer with SVD right matrix (if freeze_emb=True the matrix weights will remain fixed)
    svd_dict = model_params.get("svd", None)

    if svd_dict is not None:
        freeze_emb = model_params.get("freeze_emb",False)
        use_diag = model_params.get("use_diag",False)

        num_users = model_params["rec_model"]["num_users"]
        num_items = model_params["rec_model"]["num_items"]
        emb_size = model_params["rec_model"]["emb_size"]
        svd_cutoff = model_params.get("svd_cutoff",False)

        utility_matrix = create_utility_matrix_from_dataset(datasets['train'], num_users, num_items)

        # new (imputation)
        imputation = cfg["model"]["mean_imputation"]
        
        centered_utility_matrix = mean_imputation(utility_matrix,imputation)
        embedding_matrix = create_embedding_matrix(centered_utility_matrix, emb_size, use_diag)

        #embedding_matrix = create_embedding_matrix(utility_matrix, emb_size, use_diag)

        new_emb_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

        #initialize the item embedding matrix with the new embedding matrix 
        if svd_cutoff is not None and svd_cutoff < emb_size:
            #main_module.item_emb.weight.data[:,:svd_cutoff] = new_emb_matrix[:,:svd_cutoff]
            main_module.look_up.weight.data[:,:svd_cutoff] = new_emb_matrix[:,:svd_cutoff]
        else:
            cfg["model"].pop("svd_cutoff",None) #remove the svd_cutoff parameter if not used
            #to keep consistent with previous configs
            #main_module.item_emb.weight.data = new_emb_matrix
            main_module.look_up.weight.data = new_emb_matrix

        if freeze_emb:
            for param in main_module.item_emb.parameters():
                param.requires_grad = False
     