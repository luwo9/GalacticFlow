import numpy as np
import torch
import glob
from sklearn.decomposition import PCA
import externalize as ext
import subprocess
import flowcode
import torch.multiprocessing as mp
import copy


## Some function definitions used in the processing of the data

def rotate_galaxy_xy(galaxy, resolution=100, quant=0.75):
    """
    Rotate galaxy to align with x-axis. Creates a 2D dummy image of the galaxy and uses PCA to find the principal axis.

    Parameters
    ----------

    galaxy : np.ndarray
        Galaxy to be rotated.
    resolution : int, optional, default: 100
        Resolution of the dummy image.
    quant : float, optional, default: 0.75
        Quantile of the dummy image to be used for PCA.
    
    Returns
    -------

    galaxy_rot : np.ndarray
        Rotated galaxy.
    """
    image = np.histogram2d(galaxy[:,0], galaxy[:,1], bins=resolution)[0]
    fit = PCA(n_components=2).fit(np.argwhere(image>=np.quantile(image, quant)))
    angle = np.arctan2(*fit.components_[1])
    rot_mat = np.array([[np.cos(-angle), -np.sin(-angle)],
                        [np.sin(-angle), np.cos(-angle)]])
    galaxy_rot = np.copy(galaxy)
    galaxy_rot[:,:2] = galaxy_rot[:,:2]@rot_mat
    #Also rotate velocities
    galaxy_rot[:,3:5] = galaxy_rot[:,3:5]@rot_mat
    return galaxy_rot


#Function to operate ext_sample.py by calling multiple versions of it with subprocess.Popen, allowing for parallel sampling
#Deprecated
def parallel_sample(model, condition_or_n, GPU_nbs, split_size=300000):
    print("WARNING: This is a deprecated function. Use mp_evaluate instead.")
    n_GPUs = len(GPU_nbs)
    if isinstance(condition_or_n, int):
        condition_or_n_iter = [condition_or_n//n_GPUs]*n_GPUs
        condition_or_n_iter[-1] += condition_or_n%n_GPUs
    else:
        condition_or_n_iter = torch.split(condition_or_n, -(condition_or_n.shape[0]//-n_GPUs))
    
    #Save model and data to be used by the subprocesses
    torch.save(model, "ext_sampler/model_ext_sampler.pth")
    for i, condition_or_n in enumerate(condition_or_n_iter):
        torch.save(condition_or_n, f"ext_sampler/data_{i}.pth")

    np.save("ext_sampler/split_size_ext_sampler.npy", split_size)


    processes = []

    for i, (GPU, condition_or_n) in enumerate(zip(GPU_nbs, condition_or_n_iter)):
        process = subprocess.Popen(["python3", "ext_sample.py", f"{i}", f"{GPU}"])
        processes.append(process)

    for process in processes:
        process.wait()
    
    for process in processes:
        #If one process had an error, throw an error
        if process.poll() != 0:
            raise RuntimeError("One of the processes had an error")
    
    sample = []
    for i in range(n_GPUs):
        sample.append(torch.load(f"ext_sampler/data_{i}.pth"))
        subprocess.Popen(["rm", f"ext_sampler/data_{i}.pth"]).wait()

    sample = torch.vstack(sample)

    #Delete the model and the split_size file
    subprocess.Popen(["rm", "ext_sampler/model_ext_sampler.pth"]).wait()
    subprocess.Popen(["rm", "ext_sampler/split_size_ext_sampler.npy"]).wait()

    return sample


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

    if "x_cond" in condition and mode in ["sample", "sails"]:
        result = torch.cat([result, condition["x_cond"]], dim=1)

    return result



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




class Processor_cond():
    """
    Processor for conditional model. Made to complete several tidious tasks along the workflow of training and evaluating a conditional normalizing flow.
    Use: Initialize a Processor_cond object with the desired parameters. Then use it's methods for the desired task.
    In this workflow it is assumed, that the data will usually be a list of glaxy data arrays, of shape (N, 10+), where N is the number of stars in the galaxy.

    The workflow, intended is as follows:
    1. Read the data from the data folder with get_data.
    2. Clean the data with constrain_data.
    2.5 Choose what data is used (Conditions, which components and which galaxies) choose_subset.
    3. Prepare the data for training the flow with dist_stack and Data_to_flow.
    4. Train the flow.
    5. Sample the conditional flow with sample_conditional.
    6. Convert the sample to physical interpretation with sample_to_Data.

    The processor stores properites like mean and standard deviation of the data, or which components are learnt in log.
    This allows easily to e.g. transform a sample from the flow back to the physical interpretation.

    Parameters
    ----------

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
    
    Methods
    -------

    get_data(folder):
        Reads the data from the data folder and returns the data as a list of arrays, each containing the data of one galaxy.
    galaxy_split(Data, N_stars):
        Splits the array containing all glaxyy data into a list of arrays, each containing the data of one galaxy, the physical interpretation.
    dist_stack(Data):
        Stacks a list of arrays of glaxy data into a single array, the statistical interpretation. Can be understood as the inverse of galaxy_split.
    constrain_data(Data, M_dm_old, info):
        Does the data cleaning, based on the parameters given in the initialization.
    Data_to_flow(Data_c, log_learn):
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
    def __init__(self, percentile1=95, percentile2=95, feh_min=-8, ofe_min=-1, N_min=0, r_max=27.7):
        self.percentile1 = percentile1
        self.percentile2 = percentile2
        self.feh_min = feh_min
        self.ofe_min = ofe_min
        self.N_min = N_min
        self.r_max = r_max

    
    #Some cleaning up and clarification could be done:
    #Use lists and append as output is list anyway, clean p and clrify what this does
    #Only data reading, and calculating N_stars, M_DM, M_stars_tot and M_tot, wich is already
    #included in the data.
    def get_data(self, folder):
        """
        Reads the data from the data folder and returns the data as a list of arrays, each containing the data of one galaxy.
        Also calculates the number of stars, the dark matter mass and the stellar mass of each galaxy. The dark matter mass is read from the file name.

        Parameters
        ----------

        folder : str
            Path to the folder containing the data. Assumes a folder containing .npy files, each containing the data of one galaxy.
        
        Returns
        -------

        Data : list of arrays
            List of arrays, each containing the data of one galaxy.
        N_stars : array
            Array containing the number of stars in each galaxy.
        M_stars_s : array
            Array containing the stellar mass of each galaxy.
        M_dm : array
            Array containing the dark matter mass of each galaxy.
        """
        files = glob.glob(f"{folder}/*.npy")

        Data = []
        N_stars = np.zeros(len(files))
        M_stars = np.zeros(len(files))
        M_dm = np.zeros(len(files))

        for i, file in enumerate(files):
            galaxy = np.load(file).T

            N_stars[i] = galaxy.shape[0]
            M_dm[i] = float(file.split("_")[-1][:-4])
            M_stars[i] = np.sum(galaxy[:,9])
            Data.append(galaxy)

        return Data, N_stars, M_stars, M_dm


    #Work with Galaxy data and stack to distribution interpretation with diststack
    def galaxysplit(self, Data, N_stars):
        """
        Splits the array containing all glaxyy data into a list of arrays, each containing the data of one galaxy, the physical interpretation.
        """
        return np.split(Data,np.append(np.array([0]),np.cumsum(N_stars)))[1:-1]


    #Transform to 1 big array (interpretation as individual distribution points/samples) to feed flow
    def diststack(self, Data):
        """
        Stacks a list of arrays of glaxy data into a single array, the statistical interpretation. Can be understood as the inverse of galaxy_split.
        """
        return np.vstack(Data)


    #Data cleaning
    def constraindata(self, Data, M_dm_old, info=True):
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
        Data : list of arrays
            List of arrays, each containing the data of one galaxy.
        M_dm_old : array
            Array containing the dark matter mass of each galaxy, before the data cleaning.
        info : bool (optional) , default: True
            If True prints number of stars removed by cleaning and the number of galaxies removed by cleaning.
        
        Returns
        -------

        Data_out : list of arrays
            List of arrays, each containing the data of one galaxy, after the data cleaning.
        N_stars : array
            Updated array containing the number of stars in each galaxy.
        M_stars : array
            Updated array containing the stellar mass of each galaxy.
        M_dm_new : array
            Updated array containing the dark matter mass of each galaxy.

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
        Data_out = []
        N_stars = []
        M_stars = []
        M_dm_new = []
        N_old = 0
        for galaxy, M_dm in zip(Data, M_dm_old):
            N_old += galaxy.shape[0]

            #Constrains on stars
            #Watch out for hierachy of constraints

            #Distance
            #Get radius for a given percentile of stars
            R_max = np.percentile(np.sqrt(np.sum(galaxy[:,:3]**2, axis=1)), self.percentile1)
            #But cut at most at R_MAX_MAX(>largest galaxy in sample), dont include other structures
            R_MAX_MAX = self.r_max
            R_max = np.minimum(R_max, R_MAX_MAX)
            costrained_by_preset = R_max == R_MAX_MAX
            #Only stars within this radius
            is_validR = (np.sqrt(np.sum(galaxy[:,:3]**2, axis=1))<=(R_max))

            #If the preset cut at R_MAX_MAX was applied e.g. due to an other structure
            #Do another percentile constrain inside r=R_MAX_MAX, to exclude farout stars
            if costrained_by_preset:
                R_max2 = np.percentile(np.sqrt(np.sum(galaxy[is_validR,:3]**2, axis=1)), self.percentile2)
                is_validR = is_validR&(np.sqrt(np.sum(galaxy[:,:3]**2, axis=1))<=(R_max2))

            is_valid = is_validR

            #Metallcity
            #Ignore last 10 values of metallicity
            last_10 = np.argsort(galaxy[:,6])[-10:]
            is_valid = is_valid & (np.isin(np.arange(galaxy.shape[0]), last_10, invert=True))
            #is_valid = is_valid & (galaxy[:,6]>=np.quantile(galaxy[is_valid,6], 10e-4))&(galaxy[:,6]<=np.quantile(galaxy[is_valid,6], 0.9999))

            #Fe/H
            is_valid = is_valid & (galaxy[:,7]>=self.feh_min)
            #is_valid = is_valid & (galaxy[:,7]>=-5)&(galaxy[:,7]<=np.quantile(galaxy[is_valid,7], 0.9999))

            #O/Fe
            is_valid = is_valid & (galaxy[:,8]>=self.ofe_min)
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
            galaxy = galaxy[is_valid]

            #Calculate new number of stars
            N_star = galaxy.shape[0]

            #Constrain on galaxies (new number of stars)
            if N_star>=self.N_min:
                #Rotate the glaxy in the x-y plane so that it is horizontal, as part of the data cleaning
                #Performed here for efficiency, since some galaxies are removed by now
                galaxy = rotate_galaxy_xy(galaxy, quant=0.9)

                Data_out.append(galaxy)
                N_stars.append(N_star)
                M_stars.append(np.sum(galaxy[:,9]))
                M_dm_new.append(M_dm)
        

        N_stars = np.array(N_stars)
        M_stars = np.array(M_stars)
        M_dm_new = np.array(M_dm_new)

        if info:
            print(f"Cut out {len(Data)-len(Data_out)} of {len(Data)} galaxies, {N_old-np.sum(N_stars)} of {N_old} stars (~{(N_old-np.sum(N_stars))/N_old*100 :.0f}%).")

        return Data_out, N_stars, M_stars, M_dm_new
    

    #This function can be rewritten:
    #Components can be chosen in the data cleaning->Not so nice rather completley outsource in supplied functions, to let processing.py be static.
    #The subset can be chosen in the data cleaning-->^
    #The conditions can be moved to diststack where then also hstack is done, additional input is N_stars, M_stars, M_dm
    #In general the condition finding will also vary so it can be outsourced to a function using (galaxy, M_star, M_dm_g) and returning the Condition array as below
    #Or inputting (Data, N_stars, M_stars, M_dm) and returning the condition array, then diststack only takes additonal input.
    def choose_subset(self, Data, N_stars, M_stars, M_dm, comp_use = np.array([True]*9+[False]+[True]), cond_fn = ext.cond_M_stars, use_fn = ext.MW_like_galaxy, info=True):
        """
        Choose a subset of the data. Chosen are components, condition and galaxies.
        This is not part of the data cleaning, but rather a choice of what to be used.

        Parameters
        ----------
        Data : list of arrays
            List of arrays, each containing the data of one galaxy.
        N_stars : array
            Array containing the number of stars in each galaxy.
        M_stars : array
            Array containing the total mass of stars in each galaxy.
        M_dm : array
            Array containing the total mass of dark matter in each galaxy.
        comp_use : array of bools, optional, default: np.array([True]*9+[False]+[True])
            Array of bools, each entry corresponds to a component, if True the component is used.
            False means the component is not used. The order is: x, y, z, vx, vy, vz, Z, Fe/H, O/Fe, m_star, age.
        cond_fn : function, optional, default: externalize.cond_M_stars
            Function that takes (galaxy, N_star, M_star, M_dm_g) and returns a tuple of floats, specifiying the condition of each galaxy.
        use_fn : function, optional, default: externalize.MW_like_galaxy
            Function that takes (galaxy, N_star, M_star, M_dm_g) and returns a bool, if True the galaxy is used.
        info : bool, optional, default: True
            If true print info about the used subset of galaxies.
        
        Returns
        -------
        Data_ch : list of arrays
            List of arrays, each containing the data of one galaxy, after choosing the subset.
        N_stars_ch : array
            Array containing the number of stars in each galaxy, after choosing the subset.
        M_stars_ch : array
            Array containing the total mass of stars in each galaxy, after choosing the subset.
        M_dm_ch : array
            Array containing the total mass of dark matter in each galaxy, after choosing the subset.
        """
        Data_ch = []
        N_stars_ch = []
        M_stars_ch = []
        M_dm_ch = []

        for galaxy, N_star, M_star, M_dm_g in zip(Data, N_stars, M_stars, M_dm):
            #Choose comonents
            galaxy = galaxy[:, comp_use]
            #Choose subset
            if use_fn(galaxy, N_star, M_star, M_dm_g):
                #Choose condition properties and pad galaxy with it
                Condition = np.array([*cond_fn(galaxy, N_star, M_star, M_dm_g)])
                galaxy = np.hstack((galaxy, Condition.reshape(-1,Condition.shape[0]).repeat(galaxy.shape[0], axis=0)))

                #Now select the subset
                Data_ch.append(galaxy)
                N_stars_ch.append(N_star)
                M_stars_ch.append(M_star)
                M_dm_ch.append(M_dm_g)

        N_stars_ch = np.array(N_stars_ch)
        M_stars_ch = np.array(M_stars_ch)
        M_dm_ch = np.array(M_dm_ch)

        if info:
            #Info about galaxies choosen in this subset, not stars
            print(f"Chose {len(Data_ch)} of {len(Data)} galaxies.")
        
        return Data_ch, N_stars_ch, M_stars_ch, M_dm_ch




    def Data_to_flow(self, Data_c, transformation_functions, transformation_indices, inverse_transformations):
        """
        Converts the data to a format that can be used for training the flow.
        Namely the data is transformed to a torch tensor, the components to learn in log are transformed to log and the data is normalized such that it has mean 0 and std 1.

        Parameters
        ----------

        Data_c : array
            Array of (constrained) data, to be transformed to a format that can be used for training the flow.
        transformation_functions : list of functions
            List of functions, each function takes an array and transforms it. Maps (N, M) -> (N, M).
        transformation_indices : list of arrays
            List of arrays, each array contains the indices of the components to be transformed by the corresponding function in transformation_functions.
        inverse_transformations : list of functions
            List of functions, each function takes an array and transforms it. Maps (N, M) -> (N, M).
            The inverse of the corresponding function in transformation_functions, this is later used to transform the samples back to the physical data.

        Returns
        -------

        Data_p : torch tensor
            Tensor of the transformed data.
        """
        Data_p = np.copy(Data_c)

        #Learn components scaled with corresponding functions
        self.trf_fn_inv = inverse_transformations
        self.trf_ind = transformation_indices
        self.trf_fn = transformation_functions
        for inds, fn in zip(self.trf_ind, self.trf_fn):
            Data_p[:,inds] = fn(Data_p[:,inds])


        #Subtract mean from all values and divide by std to normalize data
        self.mu = Data_p.mean(axis=0)
        self.std = Data_p.std(axis=0)

        Data_p -= self.mu
        Data_p /= self.std

        Data_p = torch.from_numpy(Data_p).type(torch.float)
        return Data_p


    def sample_to_Data(self, raw):
        """
        Converts a flow sample back to the physical data.
        Can be understood as the inverse of Data_to_flow, see there for more details.

        Parameters
        ----------

        raw : torch tensor
            Tensor of the flow sample.
        
        Returns
        -------

        Data : array
            Array of the physical data.
        """
        std = torch.from_numpy(self.std).type(torch.float)
        mu = torch.from_numpy(self.mu).type(torch.float)
        Data = raw*std+mu
        for inds, fn in zip(self.trf_ind[::-1], self.trf_fn_inv[::-1]):
            Data[:,inds] = fn(Data[:,inds])
        return Data.numpy()

    def sample_Conditional(self, model, cond_indices, Condition, split_size=300000, GPUs=None):
        """
        Samples the conditional flow using a given condition. The condition is transformed to the format used for training the flow and then the flow is sampled.
        The sampling is done on the GPU in batches of split_size, because the GPU memory is limited. Allows for parallel sampling on multiple GPUs.

        Parameters
        ----------

        model : flowcode.NSFLow object
            The flow model to sample from.
        cond_indices : array
            Array containing the indices of the components of the condition.
        Condition : array
            Array containing the condition.
        split_size : int (optional), default: 300000
            The size of the batches the sampling is done in.
        GPUs : list of ints (optional), default: None
            List of the GPUs to use for sampling. If None, use device of model.
        
        Returns
        -------

        sample : array
            Array containing the sample of the flow for the given condition.
        """


        #Format: Contition is (N,n_cond) array cond_indices has length n_cond
        Cond_flow = np.copy(Condition)
        
        #Transform condition if scaled
        for inds, fn in zip(self.trf_ind, self.trf_fn):
            is_trf = np.isin(np.sort(cond_indices), inds)
            Cond_flow[:, is_trf] = fn(Cond_flow[:, is_trf])

        #Scale condition as used for training
        Cond_flow = (Cond_flow-(self.mu[cond_indices]))/(self.std[cond_indices])

        Cond_flow = torch.from_numpy(Cond_flow).type(torch.float)

        #Sample the flow

        sample = mp_evaluate(model, {"x_cond": Cond_flow}, mode="sample", GPUs=GPUs, split_size=split_size)

        return sample
    
    #Function to correctly evaluate the pdf of the flow
    def log_prob(self, model, data, cond_indices, GPUs, split_size=300000, inference_mode=True):
        """
        Evaluates the log probability of the flow for a given data sample.
        The evaluation is done on the GPU in batches of split_size, because the GPU memory is limited.
        This is to take into account that any scaling of the date prior to training needs to be respected in the probability density function.
        Uses the (log) Jacobian determinant of the transformation functions specified in Data_to_flow (not implemented yet). Uses the GPU the model is on.

        Parameters
        ----------

        model : flowcode.NSFLow object
            The flow model to evaluate.
        data : array
            Array containing the data sample with conditions.
        cond_indices : array
            Array containing the indices of the components of the condition.
        GPUs : list of ints
            List of the GPUs to use for evaluation.
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
        
        #device = model.parameters().__next__().device

        mask_cond = np.isin(np.arange(data.shape[1]), cond_indices)
        #Format: data is (N,n) array
        data_flow = np.copy(data)
        logdet_transform = np.zeros((data_flow.shape[0],))        

        #x_flow = trandformations(x), x_cond_flow = transformations(x_cond)
        #log p(x|x_cond) = log p_flow(x_flow|x_cond_flow) + sum_transformations(logdet_trf) , where logdet_trf is evaluated at the current x_flow at each transformation step

        for inds, fn, logdet in zip(self.trf_ind, self.trf_fn, self.trf_logdet):
            data_flow[:,inds] = fn(data_flow[:,inds])
            is_transformed = np.isin(np.arange(data_flow.shape[1]), inds)
            x_flow = data_flow[:,is_transformed & ~mask_cond]
            x_cond_flow = data_flow[:,is_transformed & mask_cond]
            #Evaluate logdet of transformation x_flow is the real input of the jacobian, i.e. jacobian matrix is len(x_flow) x len(x_flow)
            #However thaere may be some cases where the condition is needed as input e.g. transforming p(x|y) to p(r|phi)
            #with jacobian simply dr/dx = x/sqrt(x^2+y^2), which depends on y
            #The jacobian for transforming pdfs is only needed for any unconditional values as it does not need to be normalized in the condition
            #Most functions will however be logdet(x,_) where _ is not used
            if (is_transformed & ~mask_cond).any():
                logdet_transform += logdet(x_flow, x_cond_flow)

        #Now the scaling transform: x_flow = (x-mu)/std
        data_flow = (data_flow-(self.mu))/(self.std)
        #The jacobian matrix is diagonal with entries 1/std, such that the logdet is simply sum(log(1/std)) = sum(-log(std))
        #This is the same for every point in the sample.
        #The logdet for the conditional part is irrelevant as there is no normalization in the condition
        logdet_transform -= np.sum(np.log(self.std))

        #Now after all transformations thae transformed data point is:
        x_flow = data_flow[:,~mask_cond]
        x_cond_flow = data_flow[:,mask_cond]

        #Evaluate the model, use only stacks of split_size because GPU memory is limited

        x_flow = torch.from_numpy(x_flow).type(torch.float)
        x_cond_flow = torch.from_numpy(x_cond_flow).type(torch.float)
        logdet_transform = torch.from_numpy(logdet_transform).type(torch.float)

        model.eval()
        
        log_prob = mp_evaluate(model, {"x": x_flow, "x_cond": x_cond_flow}, mode="pdf", GPUs=GPUs, split_size=split_size, inference_mode=inference_mode)

        log_prob += logdet_transform

        return log_prob.numpy()

