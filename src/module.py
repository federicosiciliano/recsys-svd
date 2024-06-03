import numpy as np

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

    return embedding_matrix   