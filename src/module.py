import numpy as np

# Create utility matri from the train Dataset
def create_utility_matrix_from_dataset(dataset, num_users, num_items):
    utility_matrix_from_dataset = np.zeros((num_users+1, num_items+1)) 
    for x in dataset:
        user = x['uid']
        items = x['sid']
        ratings = x['rating']
        
        for i,item in enumerate(items):
            utility_matrix_from_dataset[user,item] = ratings[i]
        
    return utility_matrix_from_dataset[1:,1:] #FS: removed padding and fake user0: can generate errors?


# # Create utility matrix from the train DataLoader
# def create_utility_matrix(data_loader, num_users, num_items):
#     utility_matrix = np.zeros((num_users+1, num_items+1)) 
#     for batch in data_loader:
#         users = batch['uid']
#         items = batch['in_sid']
#         ratings = batch['in_rating']
        
#         for i,user in enumerate(users):
#             for j,item in enumerate(items[i]):
#                 utility_matrix[user.item(),item.item()] = ratings[i][j].item()
        
#     return utility_matrix


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