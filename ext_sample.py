#This script is meant to do the normal sample calculation from a flow.
#by launching multiple copies of it samples can be drawn in parallel.

import numpy as np
import torch
import flowcode
import sys

#The script is called with the following arguments:
#1. A number indicating the file to load the data from and to save the results to
#2. The GPU number to use

number = int(sys.argv[1])
GPU_nb = int(sys.argv[2])

device = f"cuda:{GPU_nb}"

#Load the data
condition_or_n = torch.load(f"ext_sampler/data_{number}.pth")
model = torch.load("ext_sampler/model_ext_sampler.pth", map_location=device)
split_size = int(np.load("ext_sampler/split_size_ext_sampler.npy"))

#Sample from the flow
model.eval()
sample = []
with torch.inference_mode():
    if isinstance(condition_or_n, int):
        for i in range(condition_or_n//split_size+1):
            sample_size = condition_or_n % split_size if i==0 else split_size
            if sample_size>0:
                res = (model.sample_Flow(sample_size, torch.tensor([]))).cpu()
                sample.append(res)
    else:
        for split in torch.split(condition_or_n, split_size):
            res = (model.sample_Flow(split.shape[0], split.to(device))).cpu() #Unsqueezed input
            sample.append(res)
sample = torch.vstack(sample)

#Save the sample
torch.save(sample, f"ext_sampler/data_{number}.pth")

