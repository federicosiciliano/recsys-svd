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
    get_ipython().system('pip install  --upgrade --force-reinstall git+https://github.com/federicosiciliano/easy_lightning.git')

    get_ipython().system('pip install pytorch_lightning')


# ## IMPORTS

# In[4]:


#Put all imports here
import numpy as np
from copy import deepcopy
import os
import sys
import time


from ray import tune
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig#, CheckpointConfig


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


#To import from src:

#attach the source folder to the start of sys.path
sys.path.insert(0, project_folder)
os.environ['PYTHONPATH'] = project_folder #for raytune workers


#import from src directory
import src.module as dec_module

import easy_exp, easy_rec, easy_torch #easy_data


# # MAIN

# ## Train

# ### Data

# In[7]:


cfg = easy_exp.cfg.load_configuration("config_tune")


# In[8]:
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#time.sleep(8*60*60)
print("Starting at:",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


# In[9]:


def prepare_raytune_config(cfg):
    raytune_cfg = {}
    for parameter_name, v in cfg["__exp__"]["__sweep__"]["parameters"].items():
        if "tune" in v:
            raytune_cfg[parameter_name] = getattr(tune, v["tune"]["name"])(**v["tune"]["params"])
    return raytune_cfg


# In[10]:


raytune_cfg = prepare_raytune_config(cfg)


# In[13]:


def run_config(cfg, if_exp_found=None, raytune=False):
    # exp_found
    # skip --> skip the experiment
    # load --> load the experiment
    # if not load nor skip, reruns the experiment completely

    exp_found, experiment_id = easy_exp.exp.get_set_experiment_id(cfg)
    #print("Experiment already found:", exp_found, "----> The experiment id is:", experiment_id)

    if exp_found and if_exp_found == "skip":
        #print("Skipping experiment")
        return
    
    # Save experiment (done here cause Early stopping with Tune schedulers may not run anything after training)
    easy_exp.exp.save_experiment(cfg)

    data, maps = easy_rec.preparation.prepare_rec_data(cfg)

    loaders = easy_rec.preparation.prepare_rec_dataloaders(cfg, data, maps)

    main_module = easy_rec.preparation.prepare_rec_model(cfg, maps)

    dec_module.prepare_embeddings_based_on_init(cfg, main_module, processed_data_folder)

    trainer = easy_torch.preparation.complete_prepare_trainer(cfg, experiment_id, additional_module=easy_rec, raytune=raytune)

    model = easy_torch.preparation.complete_prepare_model(cfg, main_module, additional_module=easy_rec)

    if exp_found and if_exp_found == "load":
        easy_torch.process.load_model(trainer, model, experiment_id)

    easy_torch.process.test_model(trainer, model, loaders, test_key=["val","test","train"])

    # Train the model using the prepared trainer, model, and data loaders
    easy_torch.process.train_model(trainer, model, loaders, val_key=["val","test"])

    # Early stopping with Tune schedulers may not run anything after training


# In[28]:


def run_raytune_cfg(raytune_cfg, cfg, if_exp_found=None):
    complete_cfg = deepcopy(cfg)
    complete_cfg.update(raytune_cfg)

    # save complete_cfg to a file

    run_config(complete_cfg, if_exp_found, raytune=True)


# In[15]:


# checkpoint_data = {
#     "epoch": epoch,
#     "net_state_dict": net.state_dict(),
#     "optimizer_state_dict": optimizer.state_dict(),
# }
# with tempfile.TemporaryDirectory() as checkpoint_dir:
#     data_path = Path(checkpoint_dir) / "data.pkl"
#     with open(data_path, "wb") as fp:
#         pickle.dump(checkpoint_data, fp)

#     checkpoint = Checkpoint.from_directory(checkpoint_dir)
#     train.report(
#         {"loss": val_loss / val_steps, "accuracy": correct / total},
#         checkpoint=checkpoint,
#     )


# In[16]:


max_num_epochs = cfg["model"]["trainer_params"]["max_epochs"]
scheduler = tune.schedulers.ASHAScheduler(
        metric="val_F1_@10/dataloader_idx_0",
        mode="max",
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=3)


# In[17]:


scaling_config = ScalingConfig(
    num_workers=1, use_gpu=True, resources_per_worker={"CPU": 8, "GPU": 1}
)

# run_config = RunConfig(
#     checkpoint_config=CheckpointConfig(
#         num_to_keep=2,
#         checkpoint_score_attribute="ptl/val_accuracy",
#         checkpoint_score_order="max",
#     ),
# )

# Define a TorchTrainer without hyper-parameters for Tuner
ray_trainer = TorchTrainer(
    lambda x: run_raytune_cfg(x, cfg),
    scaling_config=scaling_config,
    # run_config=run_config,
)


# In[18]:


os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0" #To avoid changing working directory
#os.environ["RAY_RESULTS_DIR"] = os.path.join(os.getcwd(),"out") #To avoid changing working directory

# In[19]:

tuner = tune.Tuner(
    ray_trainer,
    param_space={"train_loop_config": raytune_cfg},
    tune_config=tune.TuneConfig(
        # metric="val_NDCG_@10/dataloader_idx_0",
        # mode="max",
        num_samples=1000,
        scheduler=scheduler,
        time_budget_s=2*60*60, #seconds #May raise WARNING Failed to fetch metrics for
    ),
    #run_config = RunConfig(storage_path="../out/", name="ray_results")
)

results = tuner.fit()


# In[21]:


results.get_best_result(metric="val_F1_@10/dataloader_idx_0", mode="max")


# In[ ]:


#Problems with parallel execution:
# May generate the same experiment id (very unlikely)
### By default, there are N = 16^62 possible experiment ids
### L = already run experiments
### M = N - L possible experiment ids
### If generating K experiment ids concurrently, probability is sti
# TODO: reuse already run experiments, --> if using time budget, sleep for previous run-time (for fairness in comparison)


# In[ ]:

print("Ending at:",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

