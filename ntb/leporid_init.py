#!/usr/bin/env python
# coding: utf-8

# # Preparation stuff

# ## Connect to Drive

# In[20]:


connect_to_drive = False


# In[21]:


#Run command and authorize by popup --> other window
if connect_to_drive:
    from google.colab import drive
    drive.mount('/content/gdrive', force_remount=True)


# ## Install packages

# In[22]:


if connect_to_drive:
    #Install FS code
    #!pip install  --upgrade --no-deps --force-reinstall git+https://github.com/federicosiciliano/easy_lightning.git@fedsic
    get_ipython().system('pip install  --upgrade --force-reinstall git+https://github.com/PokeResearchLab/easy_lightning.git')

    get_ipython().system('pip install pytorch_lightning')


# ## IMPORTS

# In[23]:


#Put all imports here
import numpy as np
import matplotlib.pyplot as plt
#from copy import deepcopy
#import pickle
import os
import sys
#import cv2
import torch
import csv

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import jaccard_score
from scipy.sparse.linalg import eigsh


# In[24]:


import setuptools.dist


# ## Define paths

# In[25]:


#every path should start from the project folder:
project_folder = "../"
if connect_to_drive:
    project_folder = "/content/gdrive/Shareddrives/<SharedDriveName>" #Name of SharedDrive folder
    #project_folder = "/content/gdrive/MyDrive/<MyDriveName>" #Name of MyDrive folder

#Config folder should contain hyperparameters configurations
cfg_folder = os.path.join(project_folder,"cfg")

#Data folder should contain raw and preprocessed data
data_folder = os.path.join(project_folder,"data")
raw_data_folder = os.path.join(data_folder,"raw")
processed_data_folder = os.path.join(data_folder,"processed")

#Source folder should contain all the (essential) source code
source_folder = os.path.join(project_folder,"src")

#The out folder should contain all outputs: models, results, plots, etc.
out_folder = os.path.join(project_folder,"out")
img_folder = os.path.join(out_folder,"img")


# ## Import own code

# In[26]:


#To import from src:

#attach the source folder to the start of sys.path
sys.path.insert(0, project_folder)

#import from src directory
from src.module import *

import easy_exp, easy_rec, easy_torch #easy_data


# # MAIN

# ## Train

# ### Data

# In[27]:


cfg = easy_exp.cfg.load_configuration("config_rec")

for _ in cfg.sweep("model.rec_model.emb_size"):


    # In[28]:


    cfg["data_params"]["data_folder"] = raw_data_folder


    # In[29]:


    #cfg["data_params"]["test_sizes"] = [cfg["data_params.dataset_params.out_seq_len.val"],cfg["data_params.dataset_params.out_seq_len.test"]]

    data, maps = easy_rec.data_generation_utils.preprocess_dataset(**cfg["data_params"])


    # In[30]:


    #Save user and item mappings



    # In[31]:


    datasets = easy_rec.rec_torch.prepare_rec_datasets(data,**cfg["data_params"]["dataset_params"])


    # In[32]:


    cfg["data_params"]["collator_params"]["num_items"] = np.max(list(maps["sid"].values()))


    # In[33]:


    collators = easy_rec.rec_torch.prepare_rec_collators(**cfg["data_params"]["collator_params"])


    # In[34]:


    loaders = easy_rec.rec_torch.prepare_rec_data_loaders(datasets, **cfg["model"]["loader_params"], collate_fn=collators)


    # ### MODEL 

    # In[35]:


    cfg["model"]["rec_model"]["num_items"] = np.max(list(maps["sid"].values()))
    cfg["model"]["rec_model"]["num_users"] = np.max(list(maps["uid"].values()))
    cfg["model"]["rec_model"]["lookback"] = cfg["data_params"]["collator_params"]["lookback"]


    # In[36]:


    #load the default SASRec module with the specified parameters
    main_module = easy_rec.rec_torch.create_rec_model(**cfg["model"]["rec_model"])


    # In[37]:
    print(main_module.item_emb.weight.data)

    #Set the item embedding layer with SVD right matrix (if freeze_emb=True the matrix weights will remain fixed)
    useSVD = cfg["model"]["useSVD"]

    if useSVD:
        freeze_emb = cfg["model"].get("freeze_emb",False)
        cfg["model.freeze_emb"] = freeze_emb
        use_diag = cfg["model"].get("use_diag",False)
        cfg["model.use_diag"] = use_diag

        num_users = cfg["model"]["rec_model"]["num_users"]
        num_items = cfg["model"]["rec_model"]["num_items"]
        emb_size = cfg["model"]["rec_model"]["emb_size"]
        svd_cutoff = cfg["model"].get("svd_cutoff",1000000)
        
        utility_matrix = create_utility_matrix_from_dataset(datasets['train'], num_users, num_items)


        ######################## LEPORID INITIALIZATION ###############################

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

        k = 1000  # numero di vicini da considerare
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

        alpha = 0.7

        # matrice laplaciana regolarizzata
        L_reg = (1 - alpha) * L + alpha * d_max * I

        # eigendecomposition
        eigenvalues, eigenvectors = eigsh(L_reg, k=emb_size, which='SM')

        # scarto il primo autovettore (autovettore associato all'autovalore zero)
        embedding = np.matrix(eigenvectors[:, :emb_size])

        zero_vector = np.zeros((1, emb_size))
        embedding_matrix = np.concatenate((zero_vector, embedding), axis=0)
        new_emb_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

        #initialize the item embedding matrix with the new embedding matrix 
        if svd_cutoff is not None and svd_cutoff < emb_size:
            main_module.item_emb.weight.data[:,:svd_cutoff] = new_emb_matrix[:,:svd_cutoff]
        else:
            cfg["model"].pop("svd_cutoff",None) #remove the svd_cutoff parameter if not used
            #to keep consistent with previous configs
            main_module.item_emb.weight.data = new_emb_matrix
        
        if freeze_emb:
            for param in main_module.item_emb.parameters():
                param.requires_grad = False
    else:
        cfg["model"].pop("freeze_emb",None) #remove the freeze_emb parameter if not used
        cfg["model"].pop("use_diag",None) #remove the use_diag parameter if not used


    exp_found, experiment_id = easy_exp.exp.get_set_experiment_id(cfg)
    print("Experiment already found:", exp_found, "----> The experiment id is:", experiment_id)


    # In[ ]:


    if exp_found:
        continue #TODO: make the notebook/script stop here if the experiment is already found


    # In[ ]:


    trainer_params = easy_torch.preparation.prepare_experiment_id(cfg["model"]["trainer_params"], experiment_id)

    # Prepare callbacks and logger using the prepared trainer_params
    trainer_params["callbacks"] = easy_torch.preparation.prepare_callbacks(trainer_params)
    trainer_params["logger"] = easy_torch.preparation.prepare_logger(trainer_params)

    # Prepare the trainer using the prepared trainer_params
    trainer = easy_torch.preparation.prepare_trainer(**trainer_params)

    model_params = cfg["model"].copy()

    model_params["loss"] = easy_torch.preparation.prepare_loss(cfg["model"]["loss"], easy_rec.losses)

    # Prepare the optimizer using configuration from cfg
    model_params["optimizer"] = easy_torch.preparation.prepare_optimizer(**cfg["model"]["optimizer"])

    # Prepare the metrics using configuration from cfg
    model_params["metrics"] = easy_torch.preparation.prepare_metrics(cfg["model"]["metrics"], easy_rec.metrics)

    # Create the model using main_module, loss, and optimizer
    model = easy_torch.process.create_model(main_module, **model_params)


    # In[ ]:


    # Prepare the emission tracker using configuration from cfg
    #tracker = easy_torch.preparation.prepare_emission_tracker(**cfg["model"]["emission_tracker"], experiment_id=experiment_id)


    # ### Train

    # In[ ]:


    # Train the model using the prepared trainer, model, and data loaders
    easy_torch.process.train_model(trainer, model, loaders, val_key=["val","test"])


    # ### TEST

    # In[ ]:


    easy_torch.process.test_model(trainer, model, loaders)


    # In[ ]:


    # Save experiment and print the current configuration
    #save_experiment_and_print_config(cfg)
    easy_exp.exp.save_experiment(cfg)

    # Print completion message
    print("Execution completed.")
    print("######################################################################")
    print()




 

