import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.decomposition import NMF
from copy import deepcopy
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



# Center the data either by users or by items
def center_data(utility_matrix, by='center_user'):
    matrix_nan = np.where(utility_matrix == 0, np.nan, utility_matrix)
    if by =='center_user':
        user_means = np.nanmean(matrix_nan, axis=1, keepdims=True)
        centered_matrix = np.where(utility_matrix == 0, 0, utility_matrix - user_means)
    elif by == 'center_item':
        item_means = np.nanmean(matrix_nan, axis=0, keepdims=True)
        centered_matrix = np.where(utility_matrix == 0, 0, utility_matrix - item_means)
    elif by == 'center_both':
        global_mean = np.nanmean(matrix_nan)
        centered_matrix = np.where(utility_matrix == 0, 0, utility_matrix - global_mean)
    else:
        return utility_matrix
    
    return centered_matrix
       




# non-negative matrix factorization
def nmf(utility_matrix):
    nmf = NMF(n_components='auto', init='random', random_state=0)
    W = nmf.fit_transform(utility_matrix)
    H = nmf.components_

    return W, H




# fill the initial missing values with the avg of the item avg and user avg 
def user_item_imputation(utility_matrix):
    utility_matrix = np.where(utility_matrix == 0, np.nan, utility_matrix) 

    filled_matrix = utility_matrix.copy()
    user_means = np.nanmean(filled_matrix, axis=1)
    item_means = np.nanmean(filled_matrix, axis=0)
    global_mean = np.nanmean(filled_matrix)

    for i in range(filled_matrix.shape[0]):
        for j in range(filled_matrix.shape[1]):
            if np.isnan(filled_matrix[i, j]):  
                if not np.isnan(user_means[i]): 
                    user_mean = user_means[i]
                else:
                    user_mean = global_mean
                if not np.isnan(item_means[j]):
                    item_mean = item_means[j]
                else: 
                    item_mean = global_mean
                filled_matrix[i, j] = (user_mean + item_mean) / 2.0

    return filled_matrix



# apply SVD or Truncated SVD
def svd_approximation(matrix, k=None):
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    if k is not None:  
        s = s[:k]
        U = U[:, :k]
        Vt = Vt[:k, :]
    S = np.diag(s)
    return U @ S @ Vt




# initialize the matrix with the average of the user and the item and apply svd iteratively until a convergence condition
def iterative_svd(matrix, tolerance=1e-5, max_iter=100, k=None):
    utility_matrix = np.where(matrix == 0, np.nan, matrix) 

    filled_matrix = utility_matrix.copy()
    user_means = np.nanmean(filled_matrix, axis=1)
    item_means = np.nanmean(filled_matrix, axis=0)
    global_mean = np.nanmean(filled_matrix)

    for i in range(filled_matrix.shape[0]):
        for j in range(filled_matrix.shape[1]):
            if np.isnan(filled_matrix[i, j]):  
                if not np.isnan(user_means[i]): 
                    user_mean = user_means[i]
                else:
                    user_mean = global_mean
                if not np.isnan(item_means[j]):
                    item_mean = item_means[j]
                else: 
                    item_mean = global_mean
                filled_matrix[i, j] = (user_mean + item_mean) / 2.0

    for i in range(max_iter):
        print(i)
        old_matrix = deepcopy(filled_matrix)
        reconstructed_matrix = svd_approximation(old_matrix, k=k)

        filled_matrix = np.where(np.isnan(utility_matrix), reconstructed_matrix, utility_matrix)
        if np.sqrt(np.sum((filled_matrix - old_matrix)**2))/filled_matrix.size < tolerance:
            print(f"iterative_svd for initialization converged after {i + 1} iterations.")
            break
    else:
        print("Max number of iterations reached without convergence.")


    return filled_matrix

# k = 128
# new_s = deepcopy(s[:k])
# new_s = np.concatenate([new_s,np.zeros(np.min(utility_matrix.shape)-len(new_s))])
# kneedle = KneeLocator(range(1,len(new_s)+1), new_s, S=1.0, curve="convex", direction="decreasing")
# print(kneedle.knee, kneedle.elbow)
# #kneedle.plot_knee_normalized()
# TODO: use zeros or not?

def compute_knee(s):
    kneedle = KneeLocator(range(1,len(s)+1), s, S=1.0, curve="convex", direction="decreasing")
    #print(kneedle.knee, kneedle.elbow)
    knee = int(kneedle.knee)
    return knee

def get_svd_file(processed_data_folder, exp_name, experiment_id):
    svd_file = os.path.join(processed_data_folder,exp_name,experiment_id+".npz")
    return svd_file

def save_svd_results(processed_data_folder, exp_name, experiment_id, **kwargs):
    svd_file = get_svd_file(processed_data_folder, exp_name, experiment_id)
    svd_folder = os.path.join(*svd_file.split("/")[:-1])
    if not os.path.exists(svd_folder):
        os.makedirs(svd_folder)

    np.savez(svd_file, **kwargs)

def load_svd_results(processed_data_folder, svd_cfg):
    svd_file = get_svd_file(processed_data_folder, svd_cfg["__exp__"]["name"], svd_cfg["__exp__"]["experiment_id"])
    return np.load(svd_file)


def get_svd_cfg(cfg):
    svd_cfg = deepcopy(cfg)
    svd_cfg["__exp__"]["name"] = "svd"
    svd_cfg.pop("model")
    svd_cfg["data_params"].pop("collator_params")
    for key in deepcopy(svd_cfg["__exp__"]["__nosave__"]):
        if svd_cfg[key] == {}:
            svd_cfg["__exp__"]["__nosave__"].pop(key)
    return svd_cfg

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
                
# def get_set_svd(out_folder, exp_id, num_users, num_items):
#     filename = os.path.join()
#     try:
#         with open(filename, "r") as f:
#             data = json.load(f)  
#     except FileNotFoundError:
#             data = []


#     if isinstance(data, list):
#         dataset = [entry["dataset"] for entry in data]
#         if dataset_name not in dataset:
#                 utility_matrix = create_utility_matrix_from_dataset(datasets['train'], num_users, num_items)
#                 u, s, v_t = np.linalg.svd(utility_matrix)
#                 v = v_t.transpose()

#                 v_path = os.path.join(out_folder,dataset_name+"_v_matrix.npy")
#                 np.save(v_path, v)

#                 kneedle = KneeLocator(range(1,len(s)+1), s, S=1.0, curve="convex", direction="decreasing")
#                 #print(kneedle.knee, kneedle.elbow)
#                 knee = int(kneedle.knee)

#                 new_data = {"dataset": cfg["data_params"]["name"], 
#                         "knee": knee, 
#                         "v_path": os.path.join(out_folder,dataset_name+"_v_matrix.npy")
#                         }

#                 data.append(new_data)
#                 with open(filename, "w") as f:
#                         json.dump(data, f, indent=4)

#                 return knee, v
        
#         else:
#                 result = next((entry for entry in data if entry["dataset"] == dataset_name), None)
#                 current_knee = result['knee']
#                 current_v_matrix = np.load(os.path.join(out_folder,result['v_path']))

#                 return current_knee, current_v_matrix

#     else:
#         raise ValueError("The current JSON file does not contain a list.")   