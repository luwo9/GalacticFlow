#Made to train several flows, on the given galaxies, where different galaxies are left out each time
#Follows the normal workflow, as in Workflow_cond.ipynb

import flowcode
import processing
import externalize as ext

import torch
import numpy as np
import subprocess
import time
import gc


## Parameters

#Retraining mode: If True, the the models as saved by this script are loaded and the training is continued
#If False, the models are trained from scratch
retrain = False

#This number characterizes how often the training is restarted, if the training crashes
n_max_reload_on_crash = 2

#Base filename for the models savefiles
base_filename = "leavout_MttZ_MWs_CL2_24_10_512_8_lo"


#Initiate a processor to handle data
mpc = processing.Processor_cond(N_min=500, percentile2=95)

Data, N_stars, M_stars, M_dm = mpc.get_data("all_sims")

Data_const, N_stars_const, M_stars_const, M_dm_const = mpc.constraindata(Data, M_dm)

Data_sub_v, N_stars_sub_v, M_stars_sub_v, M_dm_sub_v = mpc.choose_subset(Data_const, N_stars_const, M_stars_const, M_dm_const, use_fn = ext.MW_like_galaxy, cond_fn=ext.cond_M_stars_2age_avZ)


#The GPUs to use will be proceseed to "cuda:GPU_nb"
GPU_nbs = np.array([2,3,4,5,6,7,8,9])

n_GPU_use = len(GPU_nbs)

leaveouts = [None,1,2,4,5,15] #[None,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]


class leavout_model:
    def __init__(self, leavout) -> None:
        self.leavout = leavout
        self.n_reload_on_crash = 0
        self.is_not_done = True
        self.is_supposed_to_be_running = False

    def start(self, GPU_nb):
        filename = base_filename + f"{self.leavout}"
        #Check if self.leavout is None, then leave out nothing
        leavout_fn = ext.construct_MW_like_galaxy_leavout(M_dm_sub_v[self.leavout] if self.leavout is not None else -1)
        Data_sub, N_stars_sub, M_stars_sub, M_dm_sub = mpc.choose_subset(Data_const, N_stars_const, M_stars_const, M_dm_const, use_fn = leavout_fn, cond_fn=ext.cond_M_stars_2age_avZ)

        device = f"cuda:{GPU_nb}"

        model = flowcode.NSFlow(24, 10, 4, flowcode.NSF_CL2, K=10, B=3, network=flowcode.MLP, network_args=(512,8,0.2))
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
            n_epochs = 18
            init_lr = 0.00009
            gamma = 0.998
            app = ""

        torch.save(Data_flow, "cond_trainer/data_cond_trainer.pth")
        torch.save(model, "cond_trainer/model_cond_trainer.pth")
        np.save("cond_trainer/params_cond_trainer.npy", np.append(np.array([10,11,12,13]),np.array([n_epochs,init_lr,1024,gamma])))
        np.save("cond_trainer/filename_cond_trainer.npy", filename+app)
        np.save("cond_trainer/loading_complete.npy", np.array([0]))

        process = subprocess.Popen(["python3", "cond_trainer.py", f"lo_{self.leavout}on{GPU_nb}{app}{('_'+str(self.n_reload_on_crash)) if self.n_reload_on_crash > 0 else ''}"])

        #Check if loading in cond_trainer.py was completed and can safley be overwritten, throw error if not the case after 1 minute
        for i in range(60):
            if int(np.load("cond_trainer/loading_complete.npy")) == 1:
                break
            else:
                time.sleep(1)

        self.process = process
        self.is_supposed_to_be_running = True

        self.GPU_nb = GPU_nb

    def restart(self, GPU_nb=None):
        if GPU_nb is None:
            GPU_nb = self.GPU_nb
        self.n_reload_on_crash += 1
        self.start(GPU_nb)
        self.GPU_nb = GPU_nb

    def status(self):
        return self.process.poll() if self.is_supposed_to_be_running and self.is_not_done else None





#Training of the models

#Initialize all the models i.e. their representation in this script
leavout_models = [leavout_model(leaveout) for leaveout in leaveouts]

#Load each GPU with training of one model intitially
for leavout_model, GPU in zip(leavout_models, GPU_nbs):
    leavout_model.start(GPU)

#Check the training and reload GPU with new model if one is done or with the same model if training crashed

#Check if there are models left that are not trained as desired
while np.any([leavout_model.is_not_done for leavout_model in leavout_models]):
    #Check the status of each model
    for lm in leavout_models:
        status = lm.status()
        if status is not None and (status !=1 or lm.n_reload_on_crash >= n_max_reload_on_crash):
            #Understand as done (either error we dont understand or finished or crashed too many times)
            lm.is_not_done = False
            #Start a new process if there are still leaveouts left
            new_model = [leavout_model for leavout_model in leavout_models if leavout_model.is_not_done and not leavout_model.is_supposed_to_be_running]
            if len(new_model) > 0:
                new_model = new_model[0]
                new_model.start(lm.GPU_nb)
        elif status is not None:
            #Understand as crashed with need to restart
            if lm.n_reload_on_crash < n_max_reload_on_crash:
                lm.restart()

for lm in leavout_models:
    lm.process.wait()






