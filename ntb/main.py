#!/usr/bin/env python
# coding: utf-8

# # Preparation stuff

# ## Connect to Drive

# In[1]:


connect_to_drive = False


# In[2]:


#Run command and authorize by popup --> other window
if connect_to_drive:
    from google.colab import drive
    drive.mount('/content/gdrive', force_remount=True)


# ## Install packages

# In[3]:


if connect_to_drive:
    #Install FS code
    #!pip install  --upgrade --no-deps --force-reinstall git+https://github.com/federicosiciliano/easy_lightning.git@fedsic
    get_ipython().system('pip install  --upgrade --no-deps --force-reinstall git+https://github.com/PokeResearchLab/easy_lightning.git')

    get_ipython().system('pip install pytorch_lightning')


# ## IMPORTS

# In[4]:


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
from copy import deepcopy


# In[5]:


#import setuptools.dist


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
os.environ['PYTHONPATH'] = project_folder #for raytune workers

#import from src directory
from src.module import *

import easy_exp, easy_rec, easy_torch #easy_data


# # MAIN

# ## Train

# ### Data

# In[8]:


cfg = easy_exp.cfg.load_configuration("config_rec")


# In[9]:


#---> for _ in cfg.sweep("data_params.name"):
#---> for _ in cfg.sweep("model.rec_model.emb_size"):

for _ in cfg.sweep("model.rec_model.emb_size"):
    # In[10]:


    exp_found, experiment_id = easy_exp.exp.get_set_experiment_id(cfg)
    #print("Experiment already found:", exp_found, "----> The experiment id is:", experiment_id)

    # if exp_found and if_exp_found == "skip":
    #     #print("Skipping experiment")
    #     return

    # Save experiment (done here cause Early stopping with Tune schedulers may not run anything after training)
    easy_exp.exp.save_experiment(cfg)


    # In[ ]:


    data, maps = easy_rec.preparation.prepare_rec_data(cfg)


    # In[12]:


    loaders = easy_rec.preparation.prepare_rec_dataloaders(cfg, data, maps)


    # In[ ]:


    main_module = easy_rec.preparation.prepare_rec_model(cfg, maps)


    # ### Decomposition

    # In[ ]:


    prepare_embeddings_based_on_init(cfg, main_module, processed_data_folder)


    # In[ ]:


    trainer = easy_torch.preparation.complete_prepare_trainer(cfg, experiment_id, additional_module=easy_rec)#, raytune=raytune)

    model = easy_torch.preparation.complete_prepare_model(cfg, main_module, additional_module=easy_rec)


    # In[ ]:


    # if exp_found and if_exp_found == "load":
    #     easy_torch.process.load_model(trainer, model, experiment_id)

    easy_torch.process.test_model(trainer, model, loaders, test_key=["val","test","train"])

    # Train the model using the prepared trainer, model, and data loaders
    easy_torch.process.train_model(trainer, model, loaders, val_key=["val","test"])

