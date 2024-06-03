#!/usr/bin/env python
# coding: utf-8

# # Preparation stuff

# ## Connect to Drive

# In[2]:


connect_to_drive = False


# In[3]:


#Run command and authorize by popup --> other window
if connect_to_drive:
    from google.colab import drive
    drive.mount('/content/gdrive', force_remount=True)


# ## Install packages

# In[4]:


if connect_to_drive:
    #Install FS code
    #!pip install  --upgrade --no-deps --force-reinstall git+https://github.com/federicosiciliano/easy_lightning.git@fedsic
    get_ipython().system('pip install  --upgrade --no-deps --force-reinstall git+https://github.com/PokeResearchLab/easy_lightning.git')

    get_ipython().system('pip install pytorch_lightning')


# ## IMPORTS

# In[5]:


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


# ## Define paths

# In[6]:


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

# In[7]:


#To import from src:

#attach the source folder to the start of sys.path
sys.path.insert(0, project_folder)

#import from src directory
from src.module import *

import easy_exp, easy_rec, easy_torch #easy_data


# # MAIN

# ## Train

# ### Data

# In[502]:


cfg = easy_exp.cfg.load_configuration("config_rec")


# In[503]:


cfg["data_params"]["data_folder"] = raw_data_folder


# In[504]:


#cfg["data_params"]["test_sizes"] = [cfg["data_params.dataset_params.out_seq_len.val"],cfg["data_params.dataset_params.out_seq_len.test"]]

data, maps = easy_rec.data_generation_utils.preprocess_dataset(**cfg["data_params"])


# In[505]:


#Save user and item mappings
with open(os.path.join(processed_data_folder,"user_map.csv"), "w") as f_user:
    w = csv.writer(f_user)
    w.writerows(maps['uid'].items())

with open(os.path.join(processed_data_folder,"item_map.csv"), "w") as f_item:
    w = csv.writer(f_item)
    w.writerows(maps['sid'].items())


# In[506]:


datasets = easy_rec.rec_torch.prepare_rec_datasets(data,**cfg["data_params"]["dataset_params"])


# In[507]:


cfg["data_params"]["collator_params"]["num_items"] = np.max(list(maps["sid"].values()))


# In[ ]:


collators = easy_rec.rec_torch.prepare_rec_collators(data, **cfg["data_params"]["collator_params"])


# In[508]:


loaders = easy_rec.rec_torch.prepare_rec_data_loaders(datasets, **cfg["model"]["loader_params"], collate_fn=collators)


# In[509]:


# for x in loaders["train"]:
#      break
# print(x['uid'][0])
# print(x['in_sid'][0])
# print(x['in_rating'][0])


# ### MODEL 

# In[513]:
for _ in cfg.sweep("model.rec_model.emb_size"):
    cfg["model"]["rec_model"]["num_items"] = np.max(list(maps["sid"].values()))
    cfg["model"]["rec_model"]["num_users"] = np.max(list(maps["uid"].values()))
    cfg["model"]["rec_model"]["lookback"] = cfg["data_params"]["collator_params"]["lookback"]


    # In[514]:


    #load the default SASRec module with the specified parameters
    main_module = easy_rec.rec_torch.create_rec_model(**cfg["model"]["rec_model"])
    #print(main_module)


    # In[515]:


    #Set the item embedding layer with SVD right matrix (if freeze_emb=True the matrix weights will remain fixed)

    useSVD = cfg["model"]["useSVD"]
    freeze_emb = cfg["model"]["freeze_emb"]

    if useSVD:
        num_users = cfg["model"]["rec_model"]["num_users"]
        num_items = cfg["model"]["rec_model"]["num_items"]
        emb_size = cfg["model"]["rec_model"]["emb_size"]
        
        utility_matrix = create_utility_matrix(loaders['train'], num_users, num_items)
        embedding_matrix = create_embedding_matrix(utility_matrix, emb_size)

        new_emb_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

        #initialize the item embedding matrix with the new embedding matrix 
        main_module.item_emb.weight.data = new_emb_matrix

        if freeze_emb:
            for param in main_module.item_emb.parameters():
                param.requires_grad = False


    # In[516]:


    exp_found, experiment_id = easy_exp.exp.get_set_experiment_id(cfg)
    print("Experiment already found:", exp_found, "----> The experiment id is:", experiment_id)


    # In[517]:


    if not exp_found:


        # In[518]:


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


        # In[519]:


        # Prepare the emission tracker using configuration from cfg
        tracker = easy_torch.preparation.prepare_emission_tracker(**cfg["model"]["emission_tracker"], experiment_id=experiment_id)


        # ### Train

        # In[520]:


        # Train the model using the prepared trainer, model, and data loaders
        easy_torch.process.train_model(trainer, model, loaders, tracker=tracker, val_key=["val","test"])


        # ### TEST

        # In[521]:


        easy_torch.process.test_model(trainer, model, loaders, tracker=tracker)


        # In[522]:


        # Save experiment and print the current configuration
        #save_experiment_and_print_config(cfg)
        easy_exp.exp.save_experiment(cfg)

        # Print completion message
        print("Execution completed.")
        print("######################################################################")
        print()

