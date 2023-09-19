import numpy as np
import torch
import glob
from sklearn.decomposition import PCA
import externalize as ext
import subprocess
import flowcode
import torch.multiprocessing as mp
import copy
import pandas as pd
import typing

import time
## Some function definitions used in the processing of the data

def rotate_galaxy_xy(galaxy, resolution=100, quant=0.75):
    """
    Rotate galaxy to align with x-axis. Creates a 2D dummy image of the galaxy and uses PCA to find the principal axis.

    Parameters
    ----------

    galaxy : pd.DataFrame
        Galaxy to be rotated. Must contain columns "x" and "y".
    resolution : int, optional, default: 100
        Resolution of the dummy image.
    quant : float, optional, default: 0.75
        Quantile of the dummy image to be used for PCA.
    
    Returns
    -------

    galaxy_rot : np.ndarray
        Rotated galaxy.
    """
    image = np.histogram2d(galaxy["x"], galaxy["y"], bins=resolution)[0]
    fit = PCA(n_components=2).fit(np.argwhere(image>=np.quantile(image, quant)))
    angle = np.arctan2(*fit.components_[1])
    rot_mat = np.array([[np.cos(-angle), -np.sin(-angle)],
                        [np.sin(-angle), np.cos(-angle)]])
    galaxy_rot = galaxy.copy()
    galaxy_rot[["x","y"]] = galaxy_rot[["x","y"]]@rot_mat
    #Also rotate velocities
    galaxy_rot[["vx","vy"]] = galaxy_rot[["vx","vy"]]@rot_mat
    return galaxy_rot


#For multi-gpu evaluation of the model
def _splitN(N_tot, N_per_batch):
    N_batches = N_tot//N_per_batch
    N_left = N_tot%N_per_batch
    return ([N_per_batch]*N_batches + [N_left]) if N_left != 0 else [N_per_batch]*N_batches

def _evaluate_model_on_gpu(rank, gpu_id, model_dict, model_params, condition, queue, loading_result_is_fisnished, evaluate, split_size, inference_mode):
    #condition must be dict. If evaluate is pdf, then condition must contain "x"  and "x_cond".
    #If evaluate is sample, then condition must contain "x_cond" or "N" (N in case of an unconditional model).
    #If evaluate is sails additionally "m" the Markoc chain length must be supplied.
    if evaluate not in ["pdf", "sample", "sails"]:
        raise ValueError("evaluate must be one of pdf, sample or sails")
    
    device = f"cuda:{gpu_id}"



    #Get right model params
    right_params = flowcode._get_right_hypers(model_params)

    #Load model
    model = flowcode.NSFlow(**right_params)
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()

    results_collection = []
    with torch.inference_mode(inference_mode):
        if evaluate == "pdf":
            for x_batch, x_cond_batch in zip(torch.split(condition["x"], split_size), torch.split(condition["x_cond"], split_size)):
                _, model_logdet, prior_logprob = model(x_batch.to(device), x_cond_batch.to(device))
                res = (model_logdet + prior_logprob).cpu()
                results_collection.append(res)
        elif evaluate == "sample" and "x_cond" in condition:
            for x_cond_batch in torch.split(condition["x_cond"], split_size):
                res = model.sample_Flow(len(x_cond_batch), x_cond_batch.to(device)).cpu()
                results_collection.append(res)
        elif evaluate == "sample" and "N" in condition:
            for N_batch in _splitN(condition["N"], split_size):
                res = model.sample_Flow(N_batch, torch.tensor([])).cpu()
                results_collection.append(res)
        elif evaluate == "sails" and "x_cond" in condition:
            raise NotImplementedError("Sails sampling is not implemented yet")
            for x_cond_batch in torch.split(condition["x_cond"], split_size):
                res = model.sample_sails(len(x_cond_batch), x_cond_batch.to(device), condition["m"]).cpu()#Maybe each chain in parallel? But then need super many copies of the model..
                results_collection.append(res)
        elif evaluate == "sails" and "N" in condition:
            raise NotImplementedError("Sails sampling is not implemented yet")
            for N_batch in _splitN(condition["N"], split_size):
                res = model.sample_sails(N_batch, torch.tensor([]), condition["m"]).cpu()
                results_collection.append(res)

    result = torch.cat(results_collection, dim=0)
    queue.put((rank, result))
    loading_result_is_fisnished.wait()

    

def mp_evaluate(model, condition, mode, GPUs=None ,split_size=300000, inference_mode=True):
    """
    Evaluate a model with multiprocessing on multiple GPUs.

    Parameters
    ----------
    model : flowcode.NSFlow object
        The model to evaluate.
    condition : dict
        The context for the evaluation. If mode is pdf, then condition must contain "x"  and "x_cond" (points and conditions).
        If mode is sample, then condition must contain "x_cond" or "N" (N in case of an unconditional model) (condition/ Number of points).
        If mode is sails additionally "m" the Markov chain length must be supplied.
    mode : {"pdf", "sample", "sails"}
        The mode of evaluation. If "pdf", the pdf of the model is evaluated. If "sample", samples from the model are drawn. 
        If "sails", samples from the model are drawn using sails sampling (see floccode.NSFlow.sample_sails).
    GPUs : list of ints or None (optional), deffault: None
        The GPU(number)s to use for the evaluation. If None, the model is evaluated on the current device of the model (usually CPU) see Notes.

    Returns
    -------

    result : torch.tensor
        The result of the evaluation. If mode is pdf, the result is the log pdf of the model. Otherwise the result is the samples drawn from the model.

    Notes
    -----
    If the model is on a GPU already it may take a few more seconds, because the model is copied to cpu and than back in this implementation.
    Also, if the model is on a gpu that is also in GPUs, there may be less memory available, because the model launched a second time on the same gpu in this implementation.
    """

    if mode not in ["pdf", "sample", "sails"]:
        raise ValueError("mode must be one of pdf, sample or sails")
    
    model_device = model.parameters().__next__().device
    if GPUs is not None:
        model.to("cpu")
        model_dict = model.state_dict()
        model_params = model.give_kwargs
        model.to(model_device)

        n_gpu = len(GPUs)

        #Make condition ready for distribution over GPUs: Own dict for each GPU where the values are the splits of the original ones
        condition_iter = [{} for _ in range(n_gpu)]
        for key, value in condition.items():
            if key in ["x", "x_cond"]:
                for i, split in enumerate(torch.split(value, -(len(value)//-n_gpu))):
                    condition_iter[i][key] = split
            elif key in ["N"]:
                for i, split in enumerate(_splitN(value, -(value//n_gpu))):
                    condition_iter[i][key] = split
            elif key in ["m"]:
                for i in range(n_gpu):
                    condition_iter[i][key] = value
            else:
                raise ValueError(f"condition must not contain {key}")
            

            
        #Make queue and events
        #Important Note: In case of pdf evaluations the order of the results obtained from the processes needs to be the same as the order of the points in condition["x"].
        #But the queue order depends on the time the processes finish. Therefore, we need to keep track of which process is in which position.
        #Therefore we supply an id to each process and return it with the result. The main process then sorts the results according to the id.
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        loading_result_is_fisnished = ctx.Event()

        #Start processes
        processes = []
        for i, (gpu_id, condition_batch) in enumerate(zip(GPUs, condition_iter)):
            p = ctx.Process(target=_evaluate_model_on_gpu, args=(i, gpu_id, model_dict, model_params, condition_batch, queue, loading_result_is_fisnished, mode, split_size, inference_mode))
            p.start()
            processes.append(p)

        #Get results
        results = [None for _ in range(n_gpu)]
        for _ in range(n_gpu):
            i, result = queue.get()
            #Ensure that the results are in the right order (see above)
            results[i] = result
        
        #Signal that results are loaded and the and the processes can safley terminate
        loading_result_is_fisnished.set()
        for p in processes:
            p.join()

        #Concatenate results
        result = torch.cat(results, dim=0)

    else:
        #Nicer: Write this in a function that is called here and in _evaluate_model_on_gpu...
        model.eval()

        results_collection = []
        with torch.inference_mode(inference_mode):
            if mode == "pdf":
                for x_batch, x_cond_batch in zip(torch.split(condition["x"], split_size), torch.split(condition["x_cond"], split_size)):
                    _, model_logdet, prior_logprob = model(x_batch.to(model_device), x_cond_batch.to(model_device))
                    res = (model_logdet + prior_logprob).cpu()
                    results_collection.append(res)
            elif mode == "sample" and "x_cond" in condition:
                for x_cond_batch in torch.split(condition["x_cond"], split_size):
                    res = model.sample_Flow(len(x_cond_batch), x_cond_batch.to(model_device)).cpu()
                    results_collection.append(res)
            elif mode == "sample" and "N" in condition:
                for N_batch in _splitN(condition["N"], split_size):
                    res = model.sample_Flow(N_batch, torch.tensor([])).cpu()
                    results_collection.append(res)
            elif mode == "sails" and "x_cond" in condition:
                raise NotImplementedError("Sails sampling is not implemented yet")
                for x_cond_batch in torch.split(condition["x_cond"], split_size):
                    res = model.sample_sails(len(x_cond_batch), x_cond_batch.to(model_device), condition["m"]).cpu()#Maybe each chain in parallel? But then need super many copies of the model..
                    results_collection.append(res)
            elif mode == "sails" and "N" in condition:
                raise NotImplementedError("Sails sampling is not implemented yet")
                for N_batch in _splitN(condition["N"], split_size):
                    res = model.sample_sails(N_batch, torch.tensor([]), condition["m"]).cpu()
                    results_collection.append(res)

        result = torch.cat(results_collection, dim=0)

    return result


def _custom_copy(galaxy, copy_data=True):
    """
    Meant to copy a galaxy dict, such that it can be meaningfully modified.
    """
    # if copy_data:
    #             galaxy = copy.deepcopy(galaxy)
    # else:
    #     #Make a manual shallow copy but deep copy "galaxy":galaxy_dict
    #     galaxy = {key:galaxy[key] if key!="galaxy" else copy.deepcopy(galaxy[key]) for key in galaxy.keys()}
    if copy_data:
        galaxy = copy.deepcopy(galaxy)
    
    return galaxy


class Processor():
    def __init__(self, R_max=50, feh_min=-0.3, ofe_min=-4):
        self.R_max = R_max
        self.feh_min = feh_min
        self.ofe_min = ofe_min
    

    def get_data(self, data_path):
        #y,y,z vx,vy,vz metals feh,ofe [mass] age/Gyr
        Data = np.load(data_path).T
        self.M_stars = np.sum(Data[:,9])
        return Data
    
    def constrain_data(self, Data):
        is_valid = (Data[:,7] >=self.ofe_min)&(Data[:,8]>=self.feh_min)&(np.sqrt(np.sum(Data[:,:3]**2, axis=1))<=self.R_max)
        Data_c = Data[is_valid].copy()
        self.M_stars = np.sum(Data_c[:,9])
        return Data_c[:,np.array(9*[True]+[False]+[True])]
    
    def Data_to_flow(self, Data):
        Data_p = torch.from_numpy(np.copy(Data)).type(torch.float)
        self.mu = Data_p.mean(dim=0)
        self.std = Data_p.std(dim=0)
        Data_p -= self.mu
        Data_p /= self.std
        return Data_p
    
    def sample_flow(self, model, N_samples, GPUs=None ,split_size=500000):
        model.eval()
        sample = mp_evaluate(model, {"N":N_samples}, mode="sample", GPUs=GPUs, split_size=split_size)
        return sample
    
    def sample_to_Data(self, raw):
        Data = raw*self.std+self.mu
        return Data.numpy()

#Ignore this:
# New Data format should be implemented: Passed around is only 1 variable: List of dicts, 1 for each galaxy.
# The dict is of the form {**star_properties, "galaxy":galaxy_properties}, where
# star_properties is e.g. {"x": shape (N,), "vx":shape (N,),..., "Z": shape (N,), "feh": shape (N,), "ofe": shape (N,), "mass": shape (N,), "age": shape (N,)}
# galaxy_properties is e.g. {"N_stars": int, "M_stars": float, "M_dm": float, "id": int, ...}
# Thr comeonent names are saved in a (self.)list of strings, e.g. ["x", "vx", ..., "Z", "feh", "mass", "age"]
# choose_subset then adds the condition keys to the star_properties dict e.g. M_star shape (N,), and the component names to the list of component names, e.g. "M_star"
# Any cond_fn then returns a dict e.g. {"M_star":float, "avZ":float,...} Of course use_fn and use_fn have to be adapted to this new format. No longer need dm identification (but can)
# New fn for collapsing the star_properties dict to a single array of shape (N, 10+), where 10 is the number of components. Then fn that stacks this.
# For the sampled results, use galaxysplit, N_simply hast to be known by the sampler. Then it can be expanded into a star_properties dict again.
# the galaxy_properties dict makes no sense for the sampled results, so it is not used there. (But could manually add back e.g. sim ids if comapring to data)
# Maybe also return a big star_properties dict such that the components are named, but how do get in right smaller dicts? Would need a seperate fn for that.
#Maybe just offer to return the (self.)list of component names?
#Also think about API behaviour what should be dne automatically and what not. Good idea: Give sample_galaxy an option to return rawarray or named dict. Then can do both.
#Raw array than would maybe also return (self.)list of component names this would be the best option for general_sample either (i assume)
#End of ignore this

#Data structure:
#Galaxy is a dict of the form {"stars": df_stars, "gas": df_gas, ..., "galaxy": galaxy_dict}
#Where df_... is a pandas dataframe with colums like "x", "vx", ..., "Z", "feh", "mass", "age"; galaxy_dict is a dict with keys like "M_stars", "M_dm", "id", ...
#Carried arround is a list of such dicts, one for each galaxy. (One could later implement as a class usig shared memory, but for now this is fine)
#Processer stores a list of component names, e.g. ["x", "vx", ..., "Z", "feh", "mass", "age"], How is gas ... handled? Do later with possible api changes?
#    so cannon way like dict could be better that can be saved at once.  ALL COMPS EVEN IF ALSO CONDITION!!
#Also store a list of condition names, e.g. ["M_stars", "avZ", ...] as condnames["galaxy"] with maybe also condnames["stars"] for e.g. x as condition.
#Then whenever data is passed to flow names are given (all index accesed is deprecated) and the order is checked.
#choose_subset should add a new dict (better df?) to the galaxy "parameters"/"conditions" with cond names and values, e.g. {"M_stars":float, "avZ":float,...}, diststack than handels this.
#For inference:
#Cond_sample also uses cond names and a named input, not cond_inds. Allows for option to insert or not insert cond values.
#As any inds are somewhat deprecated it should return a df for sure.
#Luckily np.split can handle the df.
#For sample_galaxy: another option if return df or galaxy (dict) with df as stars key. Conditions/Parameters could then be added if not inserted (maybe an option for that).
#    Conition is input as dict or df and then again checked for names and order (sample_conditional should probably do that)
#train_flow should also no longer take cond_inds but cond_names and then interface with data like a dataframe. (But then data_to_flow reutrns a df)
#We consider inserting conditions in get_data as bad practice, this is part of the choose_subset step, but if done:
#Add the condition names (only) to the condition names list
#No: we do it this way:
# cond_names is a dict. That has a list of names for keys "galaxy", "stars", "gas". This allows to use e.g. "x" as a star condition when learning p(v|x) for stars and p(v|z) for gas.
#In case of multiple astro objects add them acordingly in get_data and choose_subset. Choose subset must be called with the same cond fn (galaxy conds) for all astro objects.
#In future make maybe cond selection own method to avoid this special treatment.
class Processor_cond():
    """
    Processor for conditional model. Made to complete several tidious tasks along the workflow of training and evaluating a conditional normalizing flow.
    Use: Initialize a Processor_cond object. Then use it's methods for the desired task.
    
    The Data structure
    ------------------

    The data is passed around as a list of dicts, one for each galaxy. Each dict has the following keys:
    "galaxy": dict with keys like "M_stars", "M_dm", "id", ...
    And keys like:
    "stars": pandas dataframe with columns like "x", "vx", ..., "Z", "feh", "mass", "age"
    "gas": pandas dataframe with columns like "x", "vx", ..., "Z", "feh", "mass", "age"


    The workflow, intended is as follows:
    1. Read the data from the data folder with get_data.
    2. Clean the data with constrain_data.
    3 Choose what data is used (Conditions, which components and which galaxies) choose_subset.
    4. Prepare the data for training the flow with dist_stack and Data_to_flow.
    5. Train the flow.
    6. Sample the conditional flow with sample_conditional.
    7. Convert the sample to physical interpretation with sample_to_Data.

    The processor stores properites like mean and standard deviation of the data used in a normalization step before training, or which components are learnt in log.
    This allows easily to e.g. transform a sample from the flow back to the physical interpretation.

    
    Methods
    -------

    get_data(folder):
        Reads the data from the data folder and returns the data as a list of arrays, each containing the data of one galaxy.
    galaxy_split(Data, N_stars):
        Splits the array containing all glaxyy data into a list of arrays, each containing the data of one galaxy, the physical interpretation.
    dist_stack(Data):
        Stacks a list of arrays of glaxy data into a single array, the statistical interpretation. Can be understood as the inverse of galaxy_split.
    constrain_data(Galaxies):
        Does the data cleaning, based on the parameters given in the initialization.
    Data_to_flow(Galaxies_stacked, transformation_functions, transformation_components, inverse_transformations):
        Prepares the data for training the flow. Uses statistical interpretation.
    sample_to_Data(raw):
        Converts a sample from the flow back to the physical interpretation.
    sample_Conditional(model, cond_indices, Condition, device split_size):
        Samples the conditional flow.
    choose_subset(Data, N_stars, M_stars, M_dm, comp_use, cond_fn, use_fn, info):
        Specify data to be considered. Chooses which galaxies and which components are used for training.

    Atributes
    ---------

    mu : torch.tensor
        Columnwise mean of the data in statistical interpretation.
    std : torch.tensor
        Columnwise standard deviation of the data in statistical interpretation.
    log_learn : array of bools
        Array of bools, indicating which components are learnt in log.
    """
    def __init__(self):
        self.component_names = {}
        self.cond_names = {}

    

    def get_data(self, folder):
        """
        Reads the data from a data folder and returns the data as list of dicts, each containing the data of one galaxy.
        Calculates the number of stars, the dark matter mass and the stellar mass of each galaxy. The dark matter mass is read from the file name.

        Parameters
        ----------

        folder : str
            Path to the folder containing the data. Assumes a folder containing .npy files, each containing the data of one galaxy.
        
        Returns
        -------

        Galaxies : list of dicts
            List of dicts, each containing the data of one galaxy. The data is stored in two keys, "stars" and "galaxy".
            The "stars" key contains the data of the stars in the galaxy, the "galaxy" key contains global data of the galaxy,
            like the dark matter mass, the stellar mass and the number of stars.
        """
        files = glob.glob(f"{folder}/*.npy")

        Galaxies = []

        component_names = ["x", "y", "z", "vx", "vy", "vz", "Z", "feh", "ofe", "mass", "age"]
        self.component_names["stars"] = component_names#update after use_fn

        for i, file in enumerate(files):
            star_data = np.load(file).T
            M_dm = float(file.split("_")[-1][:-4])
            M_stars = np.sum(star_data[:,9])
            
            star_data = pd.DataFrame(star_data, columns=component_names)
            NIHAO_id = file.split("/")[-1].split("_")[-2]
            galaxy_globaldata = {"M_dm": M_dm, "M_stars": M_stars, "N_stars": len(star_data), "id": i, "NIHAO_id": NIHAO_id}

            galaxy = {"stars": star_data, "galaxy": galaxy_globaldata}
            Galaxies.append(galaxy)

        return Galaxies


    #Work with Galaxy data and stack to distribution interpretation with diststack
    def galaxysplit(self, Data, N_stars):
        """
        Splits the array containing all glaxyy data into a list of arrays, each containing the data of one galaxy, the physical interpretation.
        """
        return np.split(Data,np.append(np.array([0]),np.cumsum(N_stars)))[1:-1]


    #Transform to 1 big array (interpretation as individual distribution points/samples) to feed flow
    def diststack(self, Galaxies):
        """
        Stacks the Galaxies tnto a single DataFrame, the statistical interpretation. Can be understood as the inverse of galaxy_split.
        """

        #Stack conditions together in one (N_star_tot, n_cond) dataframe.
        #We use np.repeat as it is fast and efficient. (Potentially use direct indice writing? That would save np.concatenate)
        Condition_column_names = Galaxies[0]["parameters"].columns.tolist()
        n_stars = np.array([galaxy["stars"].shape[0] for galaxy in Galaxies])

        all_cond_data = np.concatenate([galaxy["parameters"].values for galaxy in Galaxies], axis=0)
        all_cond_data = np.repeat(all_cond_data, n_stars, axis=0)
        all_cond_data = pd.DataFrame(all_cond_data, columns=Condition_column_names)
        
        #Stack stars together in one (N_star_tot, n_comp) dataframe.
        all_data = [galaxy["stars"] for galaxy in Galaxies]#In such cases allow later "stars" to be given as input e.g. "gas" instead
        all_data = pd.concat(all_data, ignore_index=True)

        #Stack together
        all_data = pd.concat([all_data, all_cond_data], axis=1)
        return all_data


    #Data cleaning
    def constraindata(self, Galaxies, copy_data=True, info=True, percentile1=95, percentile2=95, feh_min=-8, ofe_min=-1, N_min=0, r_max=27.7):
        """
        Does the data cleaning, based on the parameters given in the initialization.
        Constraints stars to be used on their metallicity and distance from the center of the galaxy.

        New constrains to strars can easily be added in the format:
        is_valid = is_valid & <condition>

        The galaxies number of stars and the stellar mass of the galaxy are automatically updated.

        The total number of stars in the galaxy can be constrained, such that galaxies with less stars are excluded.

        Also again, the arrays containing the number of stars, the stellar masses and the dark matter masses are updated, so that there is one entry for each galaxy remaining.

        Parameters
        ----------
        Galaxies : list of dicts	
            List of dicts, each containing the data of one galaxy.
        copy_data : bool (optional) , default: True
            If True, the data is copied before cleaning, such that the original data is not changed.
        info : bool (optional) , default: True
            If True prints number of stars removed by cleaning and the number of galaxies removed by cleaning.
        percentile1 : float, optional, default: 95
            Percentile of the data to be used for the first percentile cut in radial distance.
        percentile2 : float, optional, default: 95
            Percentile of the data to be used for the second percentile cut in radial distance.
        feh_min : float, optional, default: -8
            Minimum [Fe/H] to be used in the data. Values below this will be excluded.
        ofe_min : float, optional, default: -1
            Minimum [O/Fe] to be used in the data. Values below this will be excluded.
        N_min : int, optional, default: 0
            Minimum number of stars to be used in the data. Galaxies with less stars will be excluded.
        
        Returns
        -------

        Galaxies_out : list of dicts
            List of dicts, each containing the data of one galaxy, after the data cleaning.

        Note
        ----

        Hierachy of constraints: R; Z; Fe/H; O/Fe
        The distance is constrained in the following way:
        The percentile1-th percentile of the stars distances is taken as the maximum distance but if this is larger than R_MAX_MAX, the maximum distance is set to R_MAX_MAX.
        R_MAX_MAX is the maximum distance of the largest galaxy in the sample expected.
        This is due to some galaxies having a large number of stars in a structure outside the main galaxy, which would lead to a too large maximum distance.
        Now if this fixed constraint was applied, there may still be outliers of the galaxies as the percentile only removd the stars from the outside structure.
        To remove these outliers, the percentile2-th percentile of the stars distances is taken as the maximum distance, for those galaxies.

        """
        Galaxies_out = []
        N_old = 0
        N_new = 0
        for galaxy in Galaxies:
            N_old += galaxy["stars"].shape[0]

            #Constrains on stars
            #Watch out for hierachy of constraints

            #Distance
            #Get radius for a given percentile of stars
            R_max = np.percentile(np.sqrt(np.sum(galaxy["stars"][["x","y","z"]]**2, axis=1)), percentile1)
            #But cut at most at R_MAX_MAX(>largest galaxy in sample), dont include other structures
            R_MAX_MAX = r_max
            R_max = np.minimum(R_max, R_MAX_MAX)
            costrained_by_preset = R_max == R_MAX_MAX
            #Only stars within this radius
            is_validR = (np.sqrt(np.sum(galaxy["stars"][["x","y","z"]]**2, axis=1))<=(R_max))

            #If the preset cut at R_MAX_MAX was applied e.g. due to an other structure
            #Do another percentile constrain inside r=R_MAX_MAX, to exclude farout stars
            if costrained_by_preset:
                R_max2 = np.percentile(np.sqrt(np.sum(galaxy["stars"][is_validR][["x","y","z"]]**2, axis=1)), percentile2)
                is_validR = is_validR&(np.sqrt(np.sum(galaxy["stars"][["x","y","z"]]**2, axis=1))<=(R_max2))

            is_valid = is_validR

            #Metallcity
            #Ignore last 10 values of metallicity
            last_10 = np.argsort(galaxy["stars"]["Z"])[-10:]
            is_valid = is_valid & (np.isin(np.arange(galaxy["stars"].shape[0]), last_10, invert=True))
            #is_valid = is_valid & (galaxy[:,6]>=np.quantile(galaxy[is_valid,6], 10e-4))&(galaxy[:,6]<=np.quantile(galaxy[is_valid,6], 0.9999))

            #Fe/H
            is_valid = is_valid & (galaxy["stars"]["feh"]>=feh_min)
            #is_valid = is_valid & (galaxy[:,7]>=-5)&(galaxy[:,7]<=np.quantile(galaxy[is_valid,7], 0.9999))

            #O/Fe
            is_valid = is_valid & (galaxy["stars"]["ofe"]>=ofe_min)
            #is_valid = is_valid & (galaxy[:,8]>=np.quantile(galaxy[is_valid,8], 10e-3))&(galaxy[:,8]<=np.quantile(galaxy[is_valid,8], 1-10e-3))


            #Metallicity
            #Wrong indices! ofe and feh are switched
            #No metallicity extreme stars
            #is_valid = (galaxy[:,7] >=self.ofe_min)&(galaxy[:,8]>=self.feh_min)
            #TEST:
            #7: -5 and 0.9999 quantile
            #8: 10e-4 quantile  and 1-10e-4 quantile
            #6: 10e-3 quantile and 0.99935 quantile
            #new:
            #


            #Apply constrains now
            galaxy = _custom_copy(galaxy, copy_data=copy_data)

            galaxy["stars"] = galaxy["stars"][is_valid]
            
            

            #Calculate new number of stars
            N_star = galaxy["stars"].shape[0]

            #Constrain on galaxies (new number of stars)
            if N_star>=N_min:
                #Rotate the glaxy in the x-y plane so that it is horizontal, as part of the data cleaning
                #Performed here for efficiency, since some galaxies are removed by now
                galaxy["stars"] = rotate_galaxy_xy(galaxy["stars"], quant=0.9)

                #Update global properties
                N_new += N_star
                galaxy["galaxy"]["N_stars"] = N_star
                galaxy["galaxy"]["M_stars"] = np.sum(galaxy["stars"]["mass"])

                Galaxies_out.append(galaxy)


        if info:
            print(f"Cut out {len(Galaxies)-len(Galaxies_out)} of {len(Galaxies)} galaxies, {N_old-N_new} of {N_old} stars (~{(N_old-N_new)/N_old*100 :.0f}%).")

        return Galaxies_out
    

    #This function can be rewritten:
    #Components can be chosen in the data cleaning->Not so nice rather completley outsource in supplied functions, to let processing.py be static.
    #The subset can be chosen in the data cleaning-->^
    #The conditions can be moved to diststack where then also hstack is done, additional input is N_stars, M_stars, M_dm
    #In general the condition finding will also vary so it can be outsourced to a function using (galaxy, M_star, M_dm_g) and returning the Condition array as below
    #Or inputting (Data, N_stars, M_stars, M_dm) and returning the condition array, then diststack only takes additonal input.
    def choose_subset(self, Galaxies, comp_use = ["x", "y", "z", "vx", "vy", "vz", "Z", "feh", "ofe", "age"], cond_fn = ext.cond_M_stars, use_fn = ext.construct_MW_like_galaxy_leavout("id",[]), pre_defined_cond=None, copy_data=True, info=True):
        """
        Choose a subset of the data. Chosen are components, condition and galaxies.
        This is not part of the data cleaning, but rather a choice of what to be used.

        Parameters
        ----------
        Galaxies : list of dicts
            Galaxy data to choose a subset from.
        comp_use : list of of strings (optional), default: ["x", "y", "z", "vx", "vy", "vz", "Z", "feh", "ofe", "age"]
            List of strings, specifying which components to use.
        cond_fn : function (optional), default: externalize.cond_M_stars
            Function that takes galaxy and returns the galaxy parameters/conditons.
            Must take galaxy as input and return dict with float values or pandas.DataFrame(dict). E.g. {"M_star": 1e11, "average_age": 5e9}.
        use_fn : function (optional), default: externalize.MW_like_galaxy
            Function that takes galaxy and returns bool, specifying if galaxy should be used.
        pre_defined_cond : list of str (optional), default: None
            List of strings, specifying which conditions are already defined in the galaxy dict, e.g. in galaxy["stars"].
            For example if one wants to learn p(v|x), then pre_defined_cond=["x"]. This will not be saved under the galaxy["parameters"].
            None translates to an empty list.
        copy_data : bool (optional), default: True
            Whether to copy the data before modifying it or not.
        info : bool (optional), default: True
            If true print info about the used subset of galaxies.
        
        Returns
        -------
        Galaxies_out : list of dicts
            Subset of the input galaxies.
        """
        if pre_defined_cond is None:
            pre_defined_cond = []
        self.cond_names["stars"] = pre_defined_cond

        Galaxies_out = []

        for galaxy in Galaxies:
            #Ccheck if galaxy should be used
            if use_fn(galaxy):
                #Copy galaxy as desired
                galaxy = _custom_copy(galaxy, copy_data=copy_data)

                #Compute galaxy conditions
                Condition = cond_fn(galaxy)
                if type(Condition)==dict:
                    Condition = pd.DataFrame(Condition)
                elif type(Condition)!=pd.DataFrame:
                    raise TypeError("cond_fn must return dict or pd.DataFrame")
                
                #For now, warn if parameters are set differently, no dont, in case it is changed maybe

                #Add conditions to galaxy, only as dict keys, diststack will add them to the data
                galaxy["parameters"] = Condition
                #Save condition names (important for e.g. ordering of components before transforming to torch tensors)
                self.cond_names["galaxy"] = Condition.columns.to_list()

                #Old:
                #Condition = np.array([*cond_fn(galaxy, N_star, M_star, M_dm_g)])
                #galaxy = np.hstack((galaxy, Condition.reshape(-1,Condition.shape[0]).repeat(galaxy.shape[0], axis=0)))

                #Choose components
                galaxy["stars"] = galaxy["stars"][comp_use]
                #And remove them from component names
                self.component_names["stars"] = [name for name in self.component_names["stars"] if name in comp_use]

                #Now add galaxy to output
                Galaxies_out.append(galaxy)

        if info:
            #Info about galaxies choosen in this subset, not stars
            print(f"Chose {len(Galaxies_out)} of {len(Galaxies)} galaxies.")
        
        return Galaxies_out




    def Data_to_flow(self, Galaxies_stacked: pd.DataFrame, transformation_functions, transformation_components, inverse_transformations, transformation_logdets=None, copy_data=True) -> pd.DataFrame:
        """
        Converts the data to a format that can be used for training the flow.
        Some manual transformations are applied, e.g. log10 for masses and the data is normalized.
        Of course this transforation needs to be inverted/respected later, which is automatically done by sample_to_Data, sample_Conditional and log_prob.

        Parameters
        ----------

        Galaxies_stacked : pd.DataFrame
            DataFrame of stacked data, to be transformed to a format that can be used for training the flow.
        transformation_functions : list of functions
            List of functions, each function takes a pd.DataFrame with columns as in transformation_components and transforms it. Maps (N, M) -> (N, M).
        transformation_components : list of lists of strings
            List of lists, each list contains the component names to be transformed by the corresponding function in transformation_functions.
        inverse_transformations : list of functions
            List of functions, each function takes a pd.DataFrame with columns as in transformation_components and transforms it. Maps (N, M) -> (N, M).
            The inverse of the corresponding function in transformation_functions, this is later used to transform the samples back to the physical data.
        transformation_logdets : list of functions (optional), default: None
            List of functions, the log jacobian determinants of the corresponding functions in transformation_functions.
            This is only needed if the pdf of the data is to be evaluated (log_prob method), not needed for sampling.
            Takes a pd.DataFrame with columns as in transformation_components and returns an array of shape (N,).

        Returns
        -------

        Galaxies_stacked : pd.DataFrame
            DataFrame of stacked data, transformed/normalized such that can be used for training the flow.
        """
        
        #This is independently implemented from _custom_copy, as this is intended for galaxy type data not stacked data and may be changed
        start = time.perf_counter()
        if copy_data:
            Galaxies_stacked = Galaxies_stacked.copy()
        #print(f"Copy data: {time.perf_counter()-start:.2f}s")
        #Learn components scaled with corresponding functions
        start = time.perf_counter()
        self.trf_fn_inv = inverse_transformations
        self.trf_comp = transformation_components
        self.trf_fn = transformation_functions
        self.trf_logdet = transformation_logdets
        for comp, fn in zip(self.trf_comp, self.trf_fn):
            Galaxies_stacked[comp] = fn(Galaxies_stacked[comp])

        #print(f"Transform data: {time.perf_counter()-start:.2f}s")
        #Subtract mean from all values and divide by std to normalize data
        start = time.perf_counter()
        self.mu = Galaxies_stacked.mean(axis=0)
        self.std = Galaxies_stacked.std(axis=0)
        #print(f"Compute mean and std: {time.perf_counter()-start:.2f}s")
        start = time.perf_counter()

        Galaxies_stacked -= self.mu
        Galaxies_stacked /= self.std
        #print(f"Normalize data: {time.perf_counter()-start:.2f}s")
        #Assure the right order of components
        start = time.perf_counter()
        Galaxies_stacked = Galaxies_stacked[self.component_names["stars"]+self.cond_names["galaxy"]]
        #print(f"Reorder data: {time.perf_counter()-start:.2f}s")
        return Galaxies_stacked
    

    def reproduce_normalization(self, Galaxies_stacked: pd.DataFrame, supress_warning=False) -> pd.DataFrame:
        """
        Reproduces the transformation described in Data_to_flow, using the stored functions, means and standard deviations.

        This should not be used for normalization of training data, but is only meant to later reproduce the normalization that was done with Data_to_flow before training.
        """
        components_given = Galaxies_stacked.columns.tolist()

        #Warn if components are given that were not normalized
        if not supress_warning:
            all_components = self.component_names["stars"]+self.cond_names["galaxy"]
            if not set(components_given) <= set(all_components):
                print(f"Warning there are components given that were never normalized. Ignoring them and normalizing the rest.")

        Galaxies_stacked = Galaxies_stacked.copy()

        for comp, fn in zip(self.trf_comp, self.trf_fn):
            is_trf = list(set(components_given) & set(comp))
            Galaxies_stacked[is_trf] = fn(Galaxies_stacked[is_trf])

        mu_use = self.mu[components_given]
        std_use = self.std[components_given]

        Galaxies_stacked = (Galaxies_stacked-mu_use)/std_use

        return Galaxies_stacked


    def sample_to_Data(self, flow_sample: pd.DataFrame) -> pd.DataFrame:
        """
        Converts a flow sample back to the physical data.
        Can be understood as the inverse of Data_to_flow, see there for more details.

        Parameters
        ----------

        flow_sample : pd.DataFrame
            DataFrame of the flow sample, to be transformed back to the physical data.
        
        Returns
        -------

        Galaxies_stacked : pd.DataFrame
            DataFrame of the flow sample, transformed back to the physical data.
        """
        #Invert normalization
        #Only transorm supplied components
        sup = flow_sample.columns
        Galaxies_stacked = flow_sample*self.std[sup]+self.mu[sup]

        #Invert additional transformations, (f*g)^-1 = g^-1 * f^-1
        for comp, fn in zip(self.trf_comp[::-1], self.trf_fn_inv[::-1]):
            comp_present = [name for name in comp if name in sup]
            Galaxies_stacked[comp_present] = fn(Galaxies_stacked[comp_present])

        return Galaxies_stacked

    def sample_Conditional(self, model, Condition:typing.Union[pd.DataFrame, dict], split_size=300000, GPUs=None, reinsert_condition="all") -> pd.DataFrame:
        """
        Samples the conditional flow using a given condition. The condition is transformed to the format used for training the flow and then the flow is sampled.
        The sampling is done on the GPU in batches of split_size, because the GPU memory is limited. Allows for parallel sampling on multiple GPUs.

        Parameters
        ----------

        model : flowcode.NSFLow object
            The flow model to sample from.
        Condition : pd.DataFrame or dict
            The (named) condition to sample from. Recomended format is DataFrame. If dict, the keys are the condition names and the values are the condition values,
            e.g. (my_Condition=){"M_star": np.array([1e10])*100, "average_age": np.array([1e9])*100}.
            If pd.DataFrame, the columns are the condition names and the rows are the condition values, e.g. pd.DataFrame(my_Condition).
        split_size : int (optional), default: 300000
            The size of the batches the sampling is done in. (=The size that will be moved to the GPU at once.)
        GPUs : list of ints (optional), default: None
            List of the GPUs to use for sampling. If None, use device of model.
        reinsert_condition : {"all", "local", "none"} (optional), default: "all"
            Whether to reinsert the condition into the sample returned.
            "all": Reinsert all condition values into the sample.
            "local": Reinsert only the condition values that are not galaxy properties but e.g. star properties (e.g. x if learning p(x|y)),
                specified as pre_defined_cond in choose_subset.
            "none": Do not reinsert any condition values into the sample.
        
        Returns
        -------

        sample : pd.DataFrame
            DataFrame containing the sample of the flow for the given condition.
        """

        #All condition names in the right order
        cond_names_in_right_order = self.cond_names["stars"] + self.cond_names["galaxy"]

        #Check the inputs
        if isinstance(Condition, dict):
            Condition = pd.DataFrame(Condition)
        elif not isinstance(Condition, pd.DataFrame):
            raise TypeError("Condition must be dict or pd.DataFrame")
        
        if not set(Condition.columns.tolist()) == set(cond_names_in_right_order):
            raise ValueError("Condition must contain all condition names and no additional ones.")
        
        #Scale as used for training
        Cond_flow = self.reproduce_normalization(Condition)

        #Make sure the order is correct (=as trained) before converting to tensor
        Cond_flow = Cond_flow[cond_names_in_right_order]
        #Convert to tensor
        Cond_flow_tensor = torch.from_numpy(Cond_flow.values).type(torch.float)

        #Sample the flow
        sample = mp_evaluate(model, {"x_cond": Cond_flow_tensor}, mode="sample", GPUs=GPUs, split_size=split_size)

        #Convert to DataFrame recover component order as used for training
        sample = sample.cpu().numpy()
        non_cond_column_names = [name for name in self.component_names["stars"] if name not in self.cond_names["stars"]] #move to @property?
        sample = pd.DataFrame(sample, columns=non_cond_column_names)

        #If desired, reinsert condition
        if reinsert_condition == "all":
            sample = pd.concat([sample, Cond_flow], axis=1)
            sample = sample[self.component_names["stars"]+self.cond_names["galaxy"]]
        elif reinsert_condition == "local":
            sample = pd.concat([sample, Cond_flow[self.cond_names["stars"]]], axis=1)
            sample = sample[self.component_names["stars"]]
        elif reinsert_condition == "none":
            sample = sample[non_cond_column_names]
        else:
            raise ValueError("reinsert_condition must be 'all', 'local' or 'none'")
        
        return sample
    
    #Function to correctly evaluate the pdf of the flow
    def log_prob(self, model, X, GPUs=None, split_size=300000, inference_mode=True):
        """
        Evaluates the log probability of the flow for a given data sample.
        The evaluation is done on the GPU in batches of split_size, because the GPU memory is limited.
        This is to take into account that any scaling of the date prior to training needs to be respected in the probability density function.
        Uses the (log) Jacobian determinant of the transformation functions specified in Data_to_flow (not implemented yet). Uses the GPU the model is on.

        Parameters
        ----------

        model : flowcode.NSFLow object
            The flow model to evaluate.
        X : pd.DataFrame or dict
            Points (With conditions) to evaluate pdf at. Recomended format is DataFrame. If dict, the keys are the component names and the values are the component values,
            e.g. (my_data=){"M_star": np.array([1e10])*100, "average_age": np.array([1e9])*100}. If pd.DataFrame, the columns are the component names and the rows are the component values,
            e.g. pd.DataFrame(my_data).
        GPUs : list of ints (optional), default: None
            List of the GPUs to use for sampling. If None, use device of model.
        split_size : int (optional), default: 300000
            The size of the batches the evaluation is done in.
        inference_mode : bool (optional), default: True
            Wheather to use torch.inference_mode() for evaluation. Will be faster, more memory efficient but does not allow gradients to be computed (e.g. dpdf(x)/dx). (?- as far as i know)
        
        Returns
        -------

        log_prob : array
            Array containing the log probability of the flow for the given data sample.
        """
        if self.trf_logdet is None:
            raise ValueError("Derivatives not specified in Data_to_flow, cannot evaluate log_prob")

        #Check the inputs
        if isinstance(X, dict):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise TypeError("X must be dict or pd.DataFrame")
        
        if not set(X.columns.tolist()) == set(self.component_names["stars"]+ self.cond_names["galaxy"]):
            raise ValueError("X must contain all component names and no additional ones.")
        

        X_flow = X.copy()
        logdet_transform = np.zeros((X_flow.shape[0],))        

        #x_flow = trandformations(x), x_cond_flow = transformations(x_cond)
        #log p(x|x_cond) = log p_flow(x_flow|x_cond_flow) + sum_transformations(logdet_trf) , where logdet_trf is evaluated at the current x_flow at each transformation step

        all_conds = self.cond_names["stars"] + self.cond_names["galaxy"]
        all_non_conds = [name for name in self.component_names["stars"] if name not in self.cond_names["stars"]]

        for comp, fn, logdet in zip(self.trf_comp, self.trf_fn, self.trf_logdet):
            X_flow[comp] = fn(X_flow[comp])

            #Order perserving set intersection
            trf_and_cond = [name for name in all_conds if name in comp]
            x_cond_flow = X_flow[trf_and_cond]
            trf_and_noncond = [name for name in all_non_conds if name in comp]
            x_flow = X_flow[trf_and_noncond]
            
            #Evaluate logdet of transformation x_flow is the real input of the jacobian, i.e. jacobian matrix is len(x_flow) x len(x_flow)
            #However thaere may be some cases where the condition is needed as input e.g. transforming p(x|y) to p(r|phi) #Thi should fail at sampling with y input as condition phi cannot be computed without x
            #with jacobian simply dr/dx = x/sqrt(x^2+y^2), which depends on y
            #The jacobian for transforming pdfs is only needed for any unconditional values as it does not need to be normalized in the condition
            #Most functions will however be logdet(x,_) where _ is not used
            if len(trf_and_cond) > 0:
                logdet_transform += logdet(x_flow, x_cond_flow)

        #Now the scaling transform: x_flow = (x-mu)/std
        X_flow = (X_flow-(self.mu))/(self.std)
        #The jacobian matrix is diagonal with entries 1/std, such that the logdet is simply sum(log(1/std)) = sum(-log(std))
        #This is the same for every point in the sample.
        #The logdet for the conditional part is irrelevant as there is no normalization in the condition
        logdet_transform -= np.sum(np.log(self.std.values))

        #Now after all transformations thae transformed data point is:
        #In the order used in the flow
        x_flow = X_flow[all_non_conds].values
        x_cond_flow = X_flow[all_conds].values

        #Evaluate the model, use only stacks of split_size because GPU memory is limited

        x_flow = torch.from_numpy(x_flow).type(torch.float)
        x_cond_flow = torch.from_numpy(x_cond_flow).type(torch.float)
        logdet_transform = torch.from_numpy(logdet_transform).type(torch.float)

        model.eval()
        
        log_prob = mp_evaluate(model, {"x": x_flow, "x_cond": x_cond_flow}, mode="pdf", GPUs=GPUs, split_size=split_size, inference_mode=inference_mode)

        log_prob += logdet_transform

        return log_prob.numpy()

    @staticmethod
    def get_array(Galaxies: "list[dict]", *keys:str):
        #We want [galaxy[key1][key2][...] for galaxy in Galaxies]
        if len(keys) == 0:
            raise ValueError("Must specify at least one key")
        result = np.zeros(len(Galaxies))
        for i, galaxy in enumerate(Galaxies):
            galaxy_ = galaxy
            for key in keys:
                galaxy_ = galaxy_[key]
            result[i] = galaxy_
        return result