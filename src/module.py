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
        
    return utility_matrix_from_dataset 


# Create utility matrix from the train DataLoader
def create_utility_matrix(data_loader, num_users, num_items):
    utility_matrix = np.zeros((num_users+1, num_items+1)) 
    for batch in data_loader:
        users = batch['uid']
        items = batch['in_sid']
        ratings = batch['in_rating']
        
        for i,user in enumerate(users):
            for j,item in enumerate(items[i]):
                utility_matrix[user.item(),item.item()] = ratings[i][j].item()
        
    return utility_matrix


def create_embedding_matrix(utility_matrix, emb_size):
    U, S, V_t = np.linalg.svd(utility_matrix)

    V = V_t.transpose()
    embedding_matrix = np.matrix(V_t[:, :emb_size])
    embedding_matrix[0,:] = 0

    return embedding_matrix   