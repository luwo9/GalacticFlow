#This is a helper script that allows to load multiple pre trained models, sample them with the right condition, and perform a specific task on all results.
#This avoinds needing to run a workflow for each model by hand.
import processing
import flowcode
import externalize as ext
import torch
import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import res_flow_vis as visual

import sys
leavout_indices = [66, 20, 88, 48, 5, 80]

model_files = ["saves/leavout_MttZ_all_CL2_24_10_512_8_lo[66, 20, 88, 48, 5].pth"]*6

GPUs = [3,4,5,6,7,8,9]

mpc = processing.Processor_cond(N_min=500, percentile2=95)

Data, N_stars, M_stars, M_dm = mpc.get_data("all_sims")

Data_const, N_stars_const, M_stars_const, M_dm_const = mpc.constraindata(Data, M_dm)

Data_sub_v, N_stars_sub_v, M_stars_sub_v, M_dm_sub_v = mpc.choose_subset(Data_const, N_stars_const, M_stars_const, M_dm_const, use_fn = ext.all_galaxies, cond_fn=ext.cond_M_stars_2age_avZ)

#results = np.zeros((5, len(leavout_indices)))
#results[0] = np.array([lo if lo is not None else np.nan for lo in leavout_indices])

for i, (leavout, model_file) in enumerate(zip(leavout_indices, model_files)):

    leavout_fn = ext.construct_all_galaxies_leavout(M_dm_sub_v[leavout_indices[:-1]])
    Data_sub, N_stars_sub, M_stars_sub, M_dm_sub = mpc.choose_subset(Data_const, N_stars_const, M_stars_const, M_dm_const, use_fn = leavout_fn, cond_fn=ext.cond_M_stars_2age_avZ)

    model = flowcode.NSFlow(24, 10, 4, flowcode.NSF_CL2, K=10, B=3, network=flowcode.MLP, network_args=(512,8,0.2))
    model = model.to("cpu")
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    
    Data_flow = mpc.Data_to_flow(mpc.diststack(Data_sub), (np.log10,), (np.array([10]),), (lambda x: 10**x,))

    cond_inds = np.array([10,11,12,13])
    Condition = Data_sub_v[leavout][:,cond_inds]

    flow_sample = mpc.sample_to_Data(mpc.sample_Conditional(model, cond_inds, Condition, split_size=int(6e5), GPUs=GPUs))

    filename = model_file.split("/")[-1].split(".")[0]
    extra = str(leavout) if leavout is not 80 else "80seen"
    visual.get_result_plots(Data_sub_v[leavout], flow_sample, label=filename+extra, format_="pdf")
    #To avoid too many open figures, we close them here
    mpl.pyplot.close("all")


    #KS_reult = visual.marginal_KS_test(Data_sub_v[leavout], flow_sample)
    #WS_reult = visual.marginal_Wasserstein_distance(Data_sub_v[leavout], flow_sample)

    #results[1,i] = KS_reult[0]
    #results[2,i] = KS_reult[1]
    #results[3,i] = WS_reult[0]
    #results[4,i] = WS_reult[1]
    print(f"Step {i} done")
    sys.stdout.flush()


#np.save("saves/aaa_MttZ_leavout_results.npy", results)

