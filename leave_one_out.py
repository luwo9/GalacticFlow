#Made to train several flows, on the given galaxies, where different galaxies are left out each time
#Follows the normal workflow, as in Workflow_cond.ipynb

import flowcode
import processing
import res_flow_vis as visual
import device_use
import externalize as ext

import torch
import numpy as np
import subprocess
import time
#For reloading modules
import importlib

#Retraining mode: If True, the the models as saved by this script are loaded and the training is continued
#If False, the models are trained from scratch
retrain = True


base_filename = "leavout_M_star_MWs_CL2_24_10_512_8_lo"

#Initiate a processor to handle data
mpc = processing.Processor_cond(N_min=500, percentile2=95)

Data, N_stars, M_stars, M_dm = mpc.get_data("all_sims")

Data_const, N_stars_const, M_stars_const, M_dm_const = mpc.constraindata(Data, M_dm)

Data_sub_v, N_stars_sub_v, M_stars_sub_v, M_dm_sub_v = mpc.choose_subset(Data_const, N_stars_const, M_stars_const, M_dm_const, use_fn = ext.MW_like_galaxy)

#The GPUs to use will be proceseed to "cuda:GPU_nb"
GPU_nbs = np.array([2,3,4,5,6,7,8,9])

n_GPU_use = len(GPU_nbs)

leaveouts = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

Processes = []

for GPU_nb, leavout in zip(GPU_nbs, leaveouts):
    filename = base_filename + f"{leavout}"
    leavout_fn = ext.construct_MW_like_galaxy_leavout(M_dm_sub_v[leavout])
    Data_sub, N_stars_sub, M_stars_sub, M_dm_sub = mpc.choose_subset(Data_const, N_stars_const, M_stars_const, M_dm_const, use_fn = leavout_fn)

    device = f"cuda:{GPU_nb}"

    with open("device_use.py", "w") as f:
        f.write(f"device_use = \"{device}\"")
    
    importlib.reload(device_use)
    importlib.reload(flowcode)

    model = flowcode.NSFlow(24, 10, 1, flowcode.NSF_CL2, K=10, B=3, network=flowcode.MLP, network_args=(512,8,0.2))
    model = model.to(device)

    if retrain:
        model.load_state_dict(torch.load(f"saves/{filename}.pth"))

    Data_flow = mpc.Data_to_flow(mpc.diststack(Data_sub), (np.log10,), (np.array([10]),), (lambda x: 10**x,))

    if retrain:
        n_epochs = 6
        init_lr = 0.000001
        gamma = 0.998
        app = "re"
    else:
        n_epochs = 10
        init_lr = 0.00009
        gamma = 0.998
        app = ""

    torch.save(Data_flow, "cond_trainer/data_cond_trainer.pth")
    torch.save(model, "cond_trainer/model_cond_trainer.pth")
    np.save("cond_trainer/params_cond_trainer.npy", np.append(np.array([10]),np.array([n_epochs,init_lr,1024,gamma])))
    np.save("cond_trainer/filename_cond_trainer.npy", filename+app)

    process = subprocess.Popen(["python3", "cond_trainer.py", f"lo_{leavout}on{GPU_nb}{app}"])

    Processes.append((process, GPU_nb))
    #Remove model from gpu memory, by moving it to cpu:
    model = model.cpu()
    #del model

cont = True

if cont:
    #Check if one of the processes is finished and start a new one on the same GPU
    #Loop over all leaveouts that are not doe yet (n_GPU_use are already done)
    for leaveout_remain in leaveouts[n_GPU_use:]:
        busy = True
        #The Model and Data could already be prepared here and later, if a gpu is free the process could be started
        #But this is not necessary, as the preparation is fast and the files loaded by cond_trainer.py would be prepared
        #and must stay unchanged for the several hours the training takes, which means that no other workflow could use them...

        #Loop over all processes, until one is finished
        while busy:
            #Wait 10 seconds before checking again
            time.sleep(10)
            for process, GPU_nb in Processes:
                if process.poll() is not None:
                    #Process is finished
                    #Thus we can start a new one on the same GPU
                    busy = False
                    Processes.remove((process, GPU_nb))
                    filename = base_filename + f"{leaveout_remain}"
                    leavout_fn = ext.construct_MW_like_galaxy_leavout(M_dm_sub_v[leaveout_remain])
                    Data_sub, N_stars_sub, M_stars_sub, M_dm_sub = mpc.choose_subset(Data_const, N_stars_const, M_stars_const, M_dm_const, use_fn = leavout_fn)

                    device = f"cuda:{GPU_nb}"

                    with open("device_use.py", "w") as f:
                        f.write(f"device_use = \"{device}\"")

                    #Reload the modules, so that the device is changed
                    importlib.reload(device_use)
                    importlib.reload(flowcode)

                    model = flowcode.NSFlow(24, 10, 1, flowcode.NSF_CL2, K=10, B=3, network=flowcode.MLP, network_args=(512,8,0.2))
                    model = model.to(device)

                    if retrain:
                        model.load_state_dict(torch.load(f"saves/{filename}.pth"))

                    Data_flow = mpc.Data_to_flow(mpc.diststack(Data_sub), (np.log10,), (np.array([10]),), (lambda x: 10**x,))

                    if retrain:
                        n_epochs = 6
                        init_lr = 0.000001
                        gamma = 0.998
                        app = "re"
                    else:
                        n_epochs = 10
                        init_lr = 0.00009
                        gamma = 0.998
                        app = ""

                    torch.save(Data_flow, "cond_trainer/data_cond_trainer.pth")
                    torch.save(model, "cond_trainer/model_cond_trainer.pth")
                    np.save("cond_trainer/params_cond_trainer.npy", np.append(np.array([10]),np.array([n_epochs,init_lr,1024,gamma])))
                    np.save("cond_trainer/filename_cond_trainer.npy", filename+app)

                    process = subprocess.Popen(["python3", "cond_trainer.py", f"lo_{leaveout_remain}on{GPU_nb}{app}"])

                    Processes.append((process, GPU_nb))
                    
                    #Remove model from gpu memory
                    model = model.cpu()
                    #del model
                    #Break out of the loop over processes, such that a new leavout is used if another process is finished
                    break

#Wait for all processes to finish, such that the subprocesses are not killed
for process, GPU_nb in Processes:
    process.wait()





