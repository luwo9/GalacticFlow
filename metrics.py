"""
Metrics for evaluating the performance of a model.
"""

import numpy as np
import pandas as pd
import API
import processing


def linear_MMD(x: np.ndarray, y: np.ndarray)->float:
    """
    Computes the linear MMD between two datasets.
    Linear MMD is defined as
    MMD = 1/n^2 sum_{i,j} (x_i - x_j)^T (x_i - x_j) + 1/m^2 sum_{i,j} (y_i - y_j)^T (y_i - y_j) - 2/nm sum_{i,j} (x_i - y_j)^T (x_i - y_j).
    """
    #Usual MMD code would be:
    # xx = pairwise_kernel(x, x)
    # yy = pairwise_kernel(y, y)
    # xy = pairwise_kernel(x, y)
    # mmd = xx.mean() + yy.mean() - 2 * xy.mean()
    #Here the kernel is <x,y>.
    #Due to the linearity of the inner product, this is equivalent to the linear MMD.
    #See https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions#Kernel_two-sample_test
    #And https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
    #This is much faster and much more memory efficient (O(n) instead of O(n^2)).
    diff = x.mean(0) - y.mean(0)
    return diff @ diff.T


def galaxy_MMD(stars1: pd.DataFrame, stars2: pd.DataFrame, processor:processing.Processor_cond, n_compare=None)->float:
    """
    Computes the linear MMD between two galaxies.
    Provides a safety wrapper around linear_MMD, to check the collumns and normalizes the data (processor.reproduce_normalization) before computation.

    If n_compare is not None, a maximum of n_compare stars will be used for the MMD computation.
    """
    if set(stars1.columns) != set(stars2.columns):
        raise ValueError("The two galaxies have different collumns.")
    
    stars2 = stars2[stars1.columns]
    stars1 = processor.reproduce_normalization(stars1).values
    stars2 = processor.reproduce_normalization(stars2).values
    if n_compare is not None:
        rng = np.random.default_rng()
        stars1 = rng.permutation(stars1)[:n_compare]
        stars2 = rng.permutation(stars2)[:n_compare]
    return linear_MMD(stars1, stars2)


def get_leavouts(model, GPUs=None, all_galaxies=None, n_compare=None, both_data=False):
    """
    For the given model, returns galaxies left out during training and a corresponding sample of the model.
    If GPUs is not None, the sample will be computed on the given GPUs.
    You can specify all_galaxies in case you evaluate multiple models that share the the same total (training+validation) galaxies.
    If n_compare is not None, the sample will be up/downsampled to n_compare stars.
    """
    if all_galaxies is None:
        all_galaxies = model.Galaxies
    leavout_key = model.leavout_train["key"]
    leavout_vals = model.leavout_train["vals"]

    leavout_galaxies = filter(lambda galaxy: galaxy["galaxy"][leavout_key] in leavout_vals, all_galaxies)
    leavout_galaxies = list(leavout_galaxies)

    #Now sample the leavout galaxies
    #If n_compare is None, sample stars as in the data
    #otherwise up/downsample to n_compare
    if n_compare is None:
        n_stars = [galaxy["stars"].shape[0] for galaxy in leavout_galaxies]
    else:
        n_stars = [n_compare for _ in leavout_galaxies]
    parameters = [galaxy["parameters"] for galaxy in leavout_galaxies]
    if both_data:
        leavout_galaxies_sample = leavout_galaxies
    else:
        leavout_galaxies_sample = model.sample_galaxy(n_stars, parameters, reinsert_conditions="galaxy", GPUs=GPUs)

    return leavout_galaxies, leavout_galaxies_sample


def get_leavout_MMDs(model:API.GalacticFlow, GPUs=None, all_galaxies=None, n_compare=None, both_data=False):
    """
    For the given model, returns the MMDs between each left out galaxy and all other galaxies.

    If GPUs is not None, the sample will be computed on the given GPUs.
    You can specify all_galaxies in case you evaluate multiple models that share the the same total (training+validation) galaxies.
    If n_compare is not None, only n_compare stars will be used for the MMD computation, data galaxies with less stars will be ignored,
    but the sample galaxy will still be upsampled to n_compare stars.
    """
    if all_galaxies is None:
        all_galaxies = model.Galaxies
    leavout_galaxies, leavout_galaxies_sample = get_leavouts(model, GPUs=GPUs, all_galaxies=all_galaxies, n_compare=n_compare, both_data=both_data)

    #Take galaxies with enough stars
    if n_compare is None:
        valid_galaxies = all_galaxies
    else:
        valid_galaxies = filter(lambda galaxy: galaxy["stars"].shape[0] >= n_compare, all_galaxies)
        valid_galaxies = list(valid_galaxies)

    #Compute the MMDs

    results = []
    conditions = model.processor.cond_names["galaxy"]
    n_conditions = len(conditions)
    n_galaxies = len(valid_galaxies)
    for data, sample in zip(leavout_galaxies, leavout_galaxies_sample):
        mmd_frame = pd.DataFrame(np.zeros((n_galaxies, n_conditions+1)), columns=["MMD"]+conditions)
        mmd_frame["MMD"] = [galaxy_MMD(sample["stars"], galaxy["stars"], model.processor, n_compare) for galaxy in valid_galaxies]
        mmd_frame[conditions] = np.array([galaxy["parameters"][conditions] for galaxy in valid_galaxies])[:,0]

        result_dict = {"id": data["galaxy"]["id"], "parameters": data["parameters"], "results": mmd_frame}
        results.append(result_dict)

    return results

def MMD_vs_params(leavout_MMDs, transform_dict=None):
    """
    Takes the output of get_leavout_MMDs (possibly from multiple models in one list) and
    returns a dataframe displaying the MMD vs the difference in each parameter.

    transform_dict is a dictionary of the form {parameter: (transform, new_name)},
    where transform is a function that transforms the parameter and new_name is the name of the transformed parameter.
    E.g. transform_dict = {"M_stars": (np.log10, "log_M_stars")}.
    """
    if transform_dict is None:
        transform_dict = {}

    all_dfs = [result["results"].copy() for result in leavout_MMDs]
    all_params = [result["parameters"].copy() for result in leavout_MMDs]
    
    
    for other, own in zip(all_dfs, all_params):
        #Transform the parameters
        for key, (transform, new_name) in transform_dict.items():
            other[key] = transform(other[key])
            own[key] = transform(own[key])
            other.rename(columns={key: new_name}, inplace=True)
            own.rename(columns={key: new_name}, inplace=True)

        #Get the column names after the transformation
        column_names = own.columns

        #Subtract the parameters to get the difference that can then be joined in one dataframe
        other[column_names] -= own[column_names].values

    #Join all the dataframes
    all_dfs = pd.concat(all_dfs, ignore_index=True)

    return all_dfs


if __name__ == "__main__":
    import glob
    import torch

    model_paths = glob.glob("saves/cross_val/model*.pth")
    ag_model = API.GalacticFlow(model_paths[0])
    ag_model.prepare()
    all_galaxies = ag_model.Galaxies

    leavout_MMDs = []
    for model in model_paths:
        model = API.GalacticFlow(model)
        #Pass all_galaxies to avoid needing to run .prepare() on every model
        leavout_MMDs += get_leavout_MMDs(model, GPUs=None, all_galaxies=all_galaxies)
    
    #torch.save(leavout_MMDs, "leavout_MMDs_data.pth")
    MMDs_vs_params = MMD_vs_params(leavout_MMDs, transform_dict={"M_stars": (np.log10, "log_M_stars")})

    MMDs_vs_params.to_csv("MMDs_vs_params.csv", index=False)