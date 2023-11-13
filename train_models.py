import API
import torch
import numpy as np
import copy
import processing
definition_model = {
    "processor": "Processor_cond",
    "processor_args": {},
    "processor_data": {"folder": "all_sims"},
    "processor_clean": {"N_min":500},
    "flow_hyper": {"n_layers":14, "dim_notcond": 10, "dim_cond": 4, "CL":"NSF_CL2", "K": 10, "B":3, "network":"MLP", "network_args":torch.tensor([128,4,0.2])},
    "subset_params": {"cond_fn": "cond_M_stars_2age_avZ", "use_fn_constructor": "construct_all_galaxies_leavout", "leavout_key": "id", "leavout_vals": [66, 20, 88, 48, 5]},
    "data_prep_args": {"transformation_functions":("np.log10",), "transformation_components":(["M_stars"],), "inverse_transformations":("10**x",), "transformation_logdets":("logdet_log10",)}
}

def get_crossval_leavouts(n_array, k):
    #choose k random numbers from 0 to n-1 (with no repeats) n/k times (with repeats)
    rng = np.random.default_rng()
    result = [rng.choice(n_array, k, replace=False) for _ in range(len(n_array)//k)]
    result = np.array(result)
    return result

if __name__ == "__main__":
    # model = API.GalacticFlow(definition_model)

    # model.prepare()

    # model.train(10,0.00009,1024,0.998, "cuda:9", info=True, update_textfile="GF_testAPI_on_9")
    # if __name__ == "__main__":
    #     API.train_GF(definition_model, 9, {"epochs": 10, "batch_size": 1024, "init_lr": 0.00009, "gamma": 0.998, "update_textfile":"GF_small_on_9"}, "saves/GF_10_64.pth", 3)

    # model.save("saves/GF_10_64.pth")


    #Modeldefs

    #This part will calculate the ids that are stillin the dataset after the cleaning
    ct_model = API.GalacticFlow(copy.deepcopy(definition_model))
    ct_model.prepare()
    existing_ids = ct_model.processor.get_array(ct_model.Galaxies, "galaxy", "id")

    #Choose the ids to leave out
    leavout_size = 5
    leavout_vals = get_crossval_leavouts(existing_ids, leavout_size)

    models = [copy.deepcopy(definition_model) for _ in range(leavout_vals.shape[0])]
    for lo, model in zip(leavout_vals, models):
        model["subset_params"]["leavout_vals"] = [int(i) for i in lo]


    #Train save and GPU settings
    train_kwargs = [{"epochs": 9, "batch_size": 1024, "init_lr": 0.00009, "gamma": 0.998, "update_textfile":f"model{list(lo)}"} for lo in leavout_vals]

    save_to = [f"saves/cv_new_temp/model{list(lo)}.pth" for lo in leavout_vals]

    GPUs = [2,5,6] #Device id's e.g. [None]*3 for 3 models at a time on CPU

    trained_models = API.train_GF(models, GPUs, train_kwargs, filenames=save_to, max_restart=3)


    