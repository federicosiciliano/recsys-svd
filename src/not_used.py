from sklearn.decomposition import NMF


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