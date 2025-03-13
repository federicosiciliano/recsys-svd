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


# ## Define paths

# In[5]:


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

# In[6]:


#import setuptools.dist


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

# In[8]:


cfg = easy_exp.cfg.load_configuration("config_init")


# In[9]:

# for _ in cfg.sweep("init.alpha"):
#     print("alpha:",cfg["init"]["alpha"])
#     for _ in cfg.sweep("init.k"):
#         print("k:",cfg["init"]["k"])

# In[13]:


exp_found, experiment_id = easy_exp.exp.get_set_experiment_id(cfg)


# In[14]:


if exp_found: raise Exception("Experiment already exists")
# if exp_found: 
#     print("Experiment already exists")
#     continue


# In[10]:


data_params = deepcopy(cfg["data_params"])
data_params["data_folder"] = raw_data_folder


# In[11]:


data, maps = easy_rec.data_generation_utils.preprocess_dataset(**data_params)


# In[12]:


datasets = easy_rec.rec_torch.prepare_rec_datasets(data,**cfg["data_params"]["dataset_params"])


# ### Compute Decomposition

# In[15]:


num_users = np.max(list(maps["uid"].values()))
num_items = np.max(list(maps["sid"].values()))

utility_matrix = create_utility_matrix_from_dataset(datasets['train'], num_users, num_items)


# In[16]:


init_params = deepcopy(cfg["init"])


# In[17]:


compute_and_save_decomposition(utility_matrix, num_items=num_items, 
                            **init_params,
                            exp_name = cfg["__exp__"]["name"], exp_id = experiment_id, save_folder=processed_data_folder)


# In[18]:


easy_exp.exp.save_experiment(cfg)

