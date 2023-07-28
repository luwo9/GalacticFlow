#The purpose of this is to allow a way of working with the model, that is not as flexible as maybe the usual workflow, but is more user friendly and easier to use.
#Therfore an abstarct model class is defined that combines the pytorch model with all the processing steps that are needed to get the model to work.
#In principle it can be adjustet as desired, if things in the workflow change, recovering full flexibility.

#This class is the main class that is used to work with the model.
#It is meant to be set with all characteristics that are specific to this model (e.g. What data it uses, what the pytorch model architecture is etc.) right from the start.
#Methods are then used for saving the whole class/state, loading it, training, sampling pdf evaluation etc.
import torch
import numpy as np
import flowcode
import processing
import externalize as ext
import time
import os
func_handle ={
    "Processor_cond": processing.Processor_cond,
    "Processor": processing.Processor,
    "NSFlow": flowcode.NSFlow,
    "NSF_CL": flowcode.NSF_CL,
    "NSF_CL2": flowcode.NSF_CL2,
    "MLP": flowcode.MLP,
    "np.log10": np.log10,
    "10**x": lambda x: 10**x,
    "MW_like_galaxy": ext.MW_like_galaxy,
    "construct_MW_like_galaxy_leavout": ext.construct_MW_like_galaxy_leavout,
    "cond_M_stars_2age_avZ": ext.cond_M_stars_2age_avZ,
    "cond_M_stars": ext.cond_M_stars,
    "all_galaxies": ext.all_galaxies,
    "construct_all_galaxies_leavout": ext.construct_all_galaxies_leavout,

}

def _handle_func(func):
    if type(func) == str:
        return func_handle[func]
    else:
        return func

class GalacticFlow():
    """
    Class that allows easily working with the models, by providing a simple and clear interface, while only sacrificing very little flexibility compared to the base workflow.
    An object of this class represents the notion of a GalacticFlow model, that has a given architecture, is or was trained on a certain dataset etc.
    Thus, all parameters that are specific to this model must be defined when creating the object. It can then be trained, saved, loaded, sampled from etc. all with the call of a single method.

    The class is meant to be used in the following way:
    1. Create an object of this class, where either:
        a) The definition is given as a dict of dicts, that contains all the information needed to define the model. (If a new model is to be tested)
        b) A file path to an already saved (abstract) model is given. (If a model is just to sampled, or retrained etc.)
    2. Call the desired method on the object. (E.g. train, sample, evaluate_pdf etc.)
    3. Save this object by simly calling the save method. Without any further information this can then be used for 1. b) again.

    Parameters
    ----------

    definition: dict of dicts or str
        The definition of the model, containing all information, either as a dict of dicts or a file path to a saved (abstract) model. See below for details.
    safe_mode: bool, (optional), default: True
        If True, the model is loaded without the use of pickle (load with weights_only=True).
        If False, arbitrary functions may be saved and loaded, but it comes with the usual risks of using pickle.

    Methods
    -------

    prepare(defintion, safe_mode=True):
        Does the advanced preparing, i.e., preparing for training. Loads and preprocesses the Data by working with the processor.
        Is called automatically by train, but can be called manually if desired.
    train(epochs, init_lr, batch_size, gamma, device):
        Trains the model for the given number of epochs, with the given learning rate, batch size and gamma on the given device.
        Keeps track of the loss and it together with the elapsed time is saved in the model. Calls prepare if not already done.
    cond_trainer_export(epochs, init_lr, batch_size, gamma, filename):
        Exports the data for training with the external cond_trainer.py script, that may be used for training with multiple GPUs in some cases.
        The data is saved in the given filename.
    general_sample(Condition, split_size=300000, GPUs=None):
        Samples from the model for the given condition, using the GPUs specified.
    sample_galaxy(self, N_stars, parameters, split_size=300000, GPUs=None):
        More simpler/physical version of general_sample, that samples a galaxy given by defined parameters. Samples as many stars as specified.
    general_pdf(self, X, split_size=300000, GPUs=None):
        Evaluates the pdf of the model for the given data X, using the GPUs specified.
    pdf_galaxy(self, galaxy, parameters, split_size=300000, GPUs=None)
        More simpler/physical version of general_pdf, that evaluates the pdf of a galaxy given by defined parameters.
    save(self, path, ensure_trained=True):
        Saves the whole GalacticFlow model to a given path.
    
    Attributes
    ----------


    Definition of a model
    ---------------------




    
    """
    def __init__(self, definition, safe_mode=True):
        #Definition is a dict of dicts that contains all the information needed to define the model or a file path to a saved (abstract) model
        #safe_mode = true means in case of a file path that it is loaded without the use of pickle (load with weights_only=True).
        #If it is intended to be loaded with safe_mode, the functions and classes must be given by a string and the corresponding function/class is used from the func_handle dict.
        #Without safe_mode arbitrary functions may be saved and loaded, but it is not safe, as it uses pickle.
        #Similarly, in case of safe_mode for every input instead of np.array use torch.tensor, such that no pickling is needed.

        #The following keys are needed for the standard model: (The Examples provided are the ones used by us for the MW model)
        #   "processor": A processor class that is used to process the data. This is needed for the training and sampling. E.g. "Processor_cond"
        #   "cond_inds": A list of indices that are used as conditional dimension, as torch.tensor. E.g. torch.tensor([10,11,12,13])
        #   "processor_args": A dict of arguments that are needed to initialize the processor. E.g. {"N_min": 500, "percentile2": 95}
        #   "processor_data": A dict with the args that are needed to get the data from the processor, contains probably only 1 key, the data folder. E.g. {"folder": "all_sims"}
        #   "flow_hyper": A dict with the hyperparmeters of the flow, see flowcode.NSFlow for details. E.g. {"n_layers":24, "dim_notcond": 10, "dim_cond": 4, "CL":"NSF_CL2", "K": 10, "B":3, "network":"MLP", "network_args":(512,8,0.2)}
        #   "subset_params": Dict for obtaining the right subset. Specify comp_use(optional), cond_fn, use_fn and train_use_fn_constructor, as well as leavout_indices (list), to leavout galaxies when training. If leavout indices is None, no galaxies are left out. E.g. {"cond_fn": "cond_M_stars_2age_avZ", "use_fn": "MW_like_galaxy", "train_use_fn_constructor": "construct_MW_like_galaxy_leavout", "leavout_indices": None}
        #   "data_prep_args": Args to processor.Data_to_flow, includes transformation_functions, transformation_indices, inverse_transformations. E.g. {"transformation_functions":("np.log10",), "transformation_indices":(torch.tensor([10]),), "inverse_transformations":("10**x",)}
        #The class will save the following things additionally:
        #   "std": The standard deviation attribute of the processor.
        #   "mean": The mean attribute of the processor.
        #   "flow_dict": The state_dict of the flow.
        #   "loss_history": A list of the loss history during training.

        self.is_loaded = type(definition) == str
        if self.is_loaded:
            #Load from file
            definition = torch.load(definition, map_location="cpu", weights_only=safe_mode)

        #Create processor
        self.processor = _handle_func(definition["processor"])
        self.processor = self.processor(**definition["processor_args"])
        self.cond_inds = definition["cond_inds"].numpy()

        #Create flow model
        flow_def = definition["flow_hyper"].copy()
        flow_def["CL"] = _handle_func(flow_def["CL"])
        flow_def["network"] = _handle_func(flow_def["network"])
        flow_def["network_args"] = (int(flow_def["network_args"][0]), int(flow_def["network_args"][1]), float(flow_def["network_args"][2]))
        self.flow = flowcode.NSFlow(**flow_def)

        #If the model is loaded from file recover  model + relavant processor attributes
        if self.is_loaded:
            self.processor.std = definition["std"].numpy() if definition["std"] is not None else None
            self.processor.mu = definition["mean"].numpy() if definition["mean"] is not None else None
            self.processor.trf_fn = tuple(_handle_func(fn) for fn in definition["data_prep_args"]["transformation_functions"])
            self.processor.trf_ind = tuple(tensor.numpy() for tensor in definition["data_prep_args"]["transformation_indices"])
            self.processor.trf_fn_inv = tuple(_handle_func(fn) for fn in definition["data_prep_args"]["inverse_transformations"])
            #logdets for pdf ... also in explanation above
            
            self.flow.load_state_dict(definition["flow_dict"])


        #Some important attributes
        self.leavout_indices = definition["subset_params"]["leavout_indices"]
        self.loss_history = definition["loss_history"].numpy() if "loss_history" in definition else []
        self.n_dim = definition["flow_hyper"]["dim_notcond"]
        self.n_cond = definition["flow_hyper"]["dim_cond"]
        #Save definition
        self.definition = definition
        self.is_prepared = False

    def prepare(self):
        #This method does the usual preparation steps for e.g. training, including loading the data, processing it, preparing it for the flow etc.
        #It is not called in the init, as it is not always needed, e.g. when only sampling from a trained model with saved std, mean etc.

        if self.is_prepared:
            print("Warning: Model already prepared, repreparing it.")

        #Load raw data
        folder = self.definition["processor_data"]["folder"]
        Data, N_stars, M_stars, M_dm = self.processor.get_data(folder)

        #Clean data
        Data_const, N_stars_const, M_stars_const, M_dm_const = self.processor.constraindata(Data, M_dm)

        #Choosing correct subset
        cond_fn = _handle_func(self.definition["subset_params"]["cond_fn"])
        use_fn = _handle_func(self.definition["subset_params"]["use_fn"])
        train_use_fn_constructor = _handle_func(self.definition["subset_params"]["train_use_fn_constructor"])
        #Check if comp_use is given
        if "comp_use" in self.definition["subset_params"].keys():
            comp_use = self.definition["subset_params"]["comp_use"]
            Data_sub_v, N_stars_sub_v, M_stars_sub_v, M_dm_sub_v = self.processor.choose_subset(Data_const, N_stars_const, M_stars_const, M_dm_const, use_fn = use_fn, cond_fn=cond_fn, comp_use=comp_use)
        else:
            Data_sub_v, N_stars_sub_v, M_stars_sub_v, M_dm_sub_v = self.processor.choose_subset(Data_const, N_stars_const, M_stars_const, M_dm_const, use_fn = use_fn, cond_fn=cond_fn)

        #Subset to train on (e.g. leave one out):
        leavout_fn = _handle_func(self.definition["subset_params"]["train_use_fn_constructor"])(M_dm_sub_v[self.leavout_indices] if self.leavout_indices is not None else -1)
        Data_sub, N_stars_sub, M_stars_sub, M_dm_sub = self.processor.choose_subset(Data_const, N_stars_const, M_stars_const, M_dm_const, use_fn = leavout_fn, cond_fn=cond_fn)

        #Prepare data for flow
        trf_fn = tuple(_handle_func(fn) for fn in self.definition["data_prep_args"]["transformation_functions"])
        trf_ind = tuple(tensor.numpy() for tensor in self.definition["data_prep_args"]["transformation_indices"])
        trf_fn_inv = tuple(_handle_func(fn) for fn in self.definition["data_prep_args"]["inverse_transformations"])
        #logdets for pdf
        Data_flow = self.processor.Data_to_flow(self.processor.diststack(Data_sub), trf_fn, trf_ind, trf_fn_inv)

        #Only save important values for now, could be extended
        self.Data_sub_v = Data_sub_v
        self.N_stars_sub_v = N_stars_sub_v
        self.M_stars_sub_v = M_stars_sub_v
        self.M_dm_sub_v = M_dm_sub_v
        self.Data_flow = Data_flow

        #Set flag
        self.is_prepared = True

    def train(self, epochs, init_lr, batch_size, gamma, device, info=False):
        """
        Trains the model for a given number of epochs here (i.e. not in the background) on the specified device.

        Parameters
        ----------

        epochs : int
            Number of epochs to train for.
        init_lr : float
            Initial learning rate.
        batch_size : int
            Batch size.
        gamma : float
            Learning rate decay factor.
        """

        #Check if model is prepared
        if not self.is_prepared:
            print("Warning: Model not prepared, preparing it now.")
            self.prepare()

        #Move to device
        self.flow.to(device)
        self.flow.train()
        #Train
        loss_history = []
        start = time.perf_counter()
        flowcode.train_flow(self.flow, self.Data_flow, self.cond_inds, epochs, lr=init_lr, batch_size=batch_size, gamma=gamma, loss_saver=loss_history)
        end = time.perf_counter()
        #Save loss history
        self.loss_history = np.array(loss_history +[end-start])

        self.flow.to("cpu")
        #gc + clear cache?
        if info:
            time_passed = (end-start)/60
            print(f"Training took about {int(time_passed/60)} hours and {int(time_passed%60)} minutes.")

    #If multithreading will work, or be used someday, this method is no longer needed- wait: if it crashes ?
    def cond_trainer_export(self, epochs, init_lr, batch_size, gamma, filename):
        """
        Export all needed quantities for cond_trainer.py. This allows training in the background.

        Parameters
        ----------
        epochs : int
            Number of epochs to train for.
        init_lr : float
            Initial learning rate.
        batch_size : int
            Batch size.
        gamma : float
            Learning rate decay factor.
        filename : str
            Filename cond trainer will save the model to. Interpret this as a tag used to later import the pytorch model back into this model.
        """
        #Check if model is prepared
        if not self.is_prepared:
            print("Warning: Model not prepared, preparing it now.")
            self.prepare()

        self.flow.train()
        #Save cond_trainer files
        torch.save(self.Data_flow, "cond_trainer/data_cond_trainer.pth")
        torch.save(self.flow, "cond_trainer/model_cond_trainer.pth")
        np.save("cond_trainer/params_cond_trainer.npy", np.append(self.cond_inds,np.array([epochs,init_lr,batch_size,gamma])))
        np.save("cond_trainer/filename_cond_trainer.npy", filename)
        np.save("cond_trainer/loading_complete.npy", np.array([0]))

    def cond_trainer_import(self, filename):
        """
        Import the model that was trained in the background with cond_trainer.py. Loaded pytorch file will be deleted(!), as the model is then saved in this model.

        Parameters
        ----------

        filename : str
            Filename used in cond_trainer_export.
        """
        real_filename = f"saves/{filename}.pth"
        
        if not os.path.isfile(real_filename):
            print("Error: File not found, cond trainer probably not finished yet.")
        else:
            self.flow.load_state_dict(torch.load(real_filename, map_location="cpu"))
            os.remove(real_filename)
            self.loss_history = np.load(f"saves/loss_{filename}.npy")
            os.remove(f"saves/loss_{filename}.npy")



    
    def general_sample(self, Condition, split_size=300000, GPUs=None):
        """
        Draws a sample from the model for a specified condition. For every condition one corresponding sample point is drawn.

        Parameters
        ----------

        Condition : array
            Condition for which to draw a sample. Must be of shape (N, self.n_cond), where N is the number of samples and n_cond the number of conditions.
        split_size : int, (optional), default: 300000
            A technical parameter. The sample is queued in chunks of size split_size for sampling. This is done to avoid memory errors on GPUs, if the sample is too large.
        GPUs : list of ints, (optional), default: None
            List of GPUs to use for sampling in parallel. If None, use device the model is currently on. The integers in the list correspond to the GPU ids.

        Returns
        -------

        sample : array
            The corrrponding sample points drawn from the model in the data space.
            Has shape (N, self.n_dim+self.n_cond), where n_dim is the dimension of the data.

        Examples
        --------

        >>> use_gpus = [0,1,2,3]
        >>> model.n_cond
        3
        >>> model.n_dim
        10
        >>> Condition = np.tile(np.array([1,2,3]), (1000,1))
        >>> sample = model.general_sample(Condition, GPUs=use_gpus)
        >>> sample.shape
        (1000, 13)
        >>> sample[0]
        array([0.,0.5,1.,0.,2.5,0.,0.,10.5,1000.,0.,1.,2.,3.])

        """
        #Maybe "recreate" option to re sample at sub_v galaxies (via Diststack and galaxysplit) for Condition?

        self.flow.eval()

        sample = self.processor.sample_Conditional(self.flow, self.cond_inds, Condition, split_size=split_size, GPUs=GPUs)

        sample = self.processor.sample_to_Data(sample)

        return sample
    
    def sample_galaxy(self, N_stars, parameters, split_size=300000, GPUs=None):
        """
        Sample a galaxy with a desired number of stars for the given parameters.
        This is a userfrienldy way to sample from the model.

        Parameters
        ----------

        N_stars : int
            Number of stars that should be sampled for this galaxy.
        parameters : array
            The parameters of the galaxy to sample. Must be of shape (self.n_cond,), where n_cond is the number of conditions.
        split_size : int, (optional), default: 300000
            A technical parameter. The sample is queued in chunks of size split_size for sampling. This is done to avoid memory errors on GPUs, if the sample is too large.
        GPUs : list of ints, (optional), default: None
            List of GPUs to use for sampling in parallel. If None, use device the model is currently on. The integers in the list correspond to the GPU ids.
        
        Returns
        -------
        galaxy : array
            The sampled galaxy. Has shape (N_stars, self.n_dim+self.n_cond), where n_dim is the dimension of the data.

        Examples
        --------

        >>> use_gpus = [0,1,2,3]
        #Model is conditional in M_star and tau50
        >>> model.n_cond
        2
        #Model is learnt on 10D data space (x,y,z,vx,...)
        >>> model.n_dim
        10
        #Sample a galaxy with total stellar mass of 10^10 M_sun and tau50 of 6 Gyr
        >>> parameters = np.array([10^10,6])
        #Sample 10^6 stars for this galaxy
        >>> galaxy = model.sample_galaxy(10^6, parameters, GPUs=use_gpus)
        >>> galaxy.shape
        (10^6, 12)
        >>> galaxy[0]
        array([0.,0.5,1.,0.,2.5,0.,0.,10.5,1000.,0.,10^10,6])
        """
        galaxy = self.general_sample(np.tile(parameters, (N_stars,1)), split_size=split_size, GPUs=GPUs)

        return galaxy
    
    def general_pdf(self, X, split_size=300000, GPUs=None):
        """
        Evaluate the log probability density function at a given set of points.

        Parameters
        ----------
        X : array
            Points on which to evaluate the pdf. Must be of shape (N, self.n_dim+self.n_cond), where N is the number of points, n_dim the dimension of the data and n_cond the number of conditions.
        split_size : int, (optional), default: 300000
            A technical parameter. The sample is queued in chunks of size split_size for sampling. This is done to avoid memory errors on GPUs, if the sample is too large.
        GPUs : list of ints, (optional), default: None
            List of GPUs to use for sampling in parallel. If None, use device the model is currently on. The integers in the list correspond to the GPU ids.
        
        Returns
        -------
        pdf : array
            The log probability density function evaluated at the given points. Has shape (N,).    
        
        Examples
        --------

        >>> use_gpus = [0,1,2,3]
        >>> model.n_cond
        3
        >>> model.n_dim
        10
        >>> Conditions = np.tile(np.array([1,2.2,3]), (4,1))
        >>> Data = np.tile(np.array([1,2,3,4,2,6,5,3.5,9,1]), (4,1))
        >>> X = np.concatenate((Data, Conditions), axis=1)
        >>> pdf = model.general_pdf(X, GPUs=use_gpus)
        >>> pdf.shape
        (4,)
        >>> pdf
        array([-1.2,-1.5,1.8,-1.9])
        """
        if GPUs is not None:
            print("Warning: Multiple GPU support not implemented yet. Using current device.")

        self.flow.eval()
        pdf = self.processor.log_prob(self.flow, X, self.cond_inds, split_size=split_size)

        return pdf
    
    def pdf_galaxy(self, galaxy, parameters, split_size=300000, GPUs=None):
        """
        Evaluate the log probability density function for a galaxy with a given set of parameters.
        This is a userfrienldy way to evaluate the pdf.

        Parameters
        ----------

        galaxy : array
            The points on which to evaluate the pdf. Must be of shape (N, self.n_dim), where N is the number of points and n_dim the dimension of the data.
        parameters : array
            The parameters of the galaxy to evaluate the pdf for. Must be of shape (self.n_cond,), where n_cond is the number of conditions.
        split_size : int, (optional), default: 300000
            A technical parameter. The sample is queued in chunks of size split_size for sampling. This is done to avoid memory errors on GPUs, if the sample is too large.
        GPUs : list of ints, (optional), default: None
            List of GPUs to use for sampling in parallel. If None, use device the model is currently on. The integers in the list correspond to the GPU ids.
        
        Returns
        -------
        pdf : array
            The log probability density function evaluated at the given points. Has shape (N,).
        
        Examples
        --------

        >>> use_gpus = [0,1,2,3]
        #Model is conditional in M_star and tau50
        >>> model.n_cond
        2
        #Model is learnt on 10D data space (x,y,z,vx,...)
        >>> model.n_dim
        10
        #Sample a galaxy with total stellar mass of 10^10 M_sun and tau50 of 6 Gyr
        >>> parameters = np.array([10^10,6])
        #Evaluate the pdf for this galaxy
        >>> galaxy = np.array([[0.,0.5,1.,0.,2.5,0.,0.,10.5,1000.,0.],[1.,2.5,1.,0.5,2.5,10.,0.,10.5,150.,0.]])
        >>> pdf = model.pdf_galaxy(galaxy, parameters, GPUs=use_gpus)
        >>> pdf.shape
        (2,)
        >>> pdf
        array([10,-3])
        >>> galaxy = model.sample_galaxy(10^6, parameters, GPUs=use_gpus)
        >>> pdf = model.pdf_galaxy(galaxy, parameters, GPUs=use_gpus)
        >>> pdf.shape
        (10^6,)
        >>> pdf
        array([10,-3,0.5,...])
        """

        pdf = self.general_pdf(np.concatenate((galaxy, np.tile(parameters, (galaxy.shape[0],1))), axis=1), split_size=split_size, GPUs=GPUs)

        return pdf
    
    def save(self, path, ensure_trained=True):
        """
        Save this model to a file. This will not just save the pytorch model, but the whole state of the GalacticFlow model/object see Notes.
        This file will later be used to load again at initialization.

        Parameters
        ----------

        path : str
            The path to save the model to.
        ensure_trained : bool, (optional), default: True
            If True only allow saving models that make sense to be saved (i.e. trained/prepared at some point).
            If False the model will be saved even if never trained or prepared.
            This is not recomended, because this state is essentially given by the definition of the model, such that there is no need really to save this state.
        
        Notes
        -----
        This function will save information like the conditional indices chosen or the processing related quantities that are required to load the model again.
        More precisely, the following information is saved:
        - The definition of the model  that has beed used to first initialize the model. (Dimensions, number of conditions, etc.)
        - The state_dict of the pytorch model (normalizing flow).
        - The state of the processor (mu and std), that otherwise would have to be computed by self.prepare() before inference.
        - The loss history of the model (including the training time).

        Examples

        >>> model.n_cond
        2
        >>> model.n_dim
        10
        >>> model.save("my_model.pth")
        >>> model = GalacticFlow("my_model.pth")
        >>> model.n_cond
        2
        >>> model.n_dim
        10
        """
        if self.loss_history == []:
            untrained = True
        else:
            untrained = False
        
        if hasattr(self.processor, "mu"):
            unprepared = False
        else:
            unprepared = True

        if ensure_trained:
            if unprepared:
                raise RuntimeError("Model has never been prepared. Cannot save. If you want to save anyway, set ensure_trained=False.")
            elif untrained:
                raise RuntimeError("Model has never been trained. Cannot save. If you want to save anyway, set ensure_trained=False.")
            else:
                #Everything is okay
                mu = torch.from_numpy(self.processor.mu)
                std = torch.from_numpy(self.processor.std)
        else:
            if unprepared:
                print("Warning: Model has never been prepared. Saving anyway with mu=None and std=None.")
                mu = None
                std = None
                #Setting to None will ensure that it is saved, but can not be used for inference by accident
                #But this may not be catched in inference if a custom Processor Class does ,e.g., define mu and std as 0 and 1 in __init__... maybe some better way to do this?
            elif untrained:
                print("Warning: Model has never been trained. Saving anyway.")
                mu = torch.from_numpy(self.processor.mu)
                std = torch.from_numpy(self.processor.std)
            else:
                #Everything is okay
                mu = torch.from_numpy(self.processor.mu)
                std = torch.from_numpy(self.processor.std)

        #Use definition of model to save + extra keywords (see __init__)
        save_dict = self.definition.copy()

        save_dict["mean"] = mu
        save_dict["std"] = std
        #watch flow's device? but loading does already so maybe not that bad
        save_dict["flow_dict"] = self.flow.state_dict()
        save_dict["loss_history"] = torch.tensor(self.loss_history)

        torch.save(save_dict, path)


#Example for a definition dictionary
#We recommend to use this as a template for your own models. Simply copy and paste and change the values accordingly.
example_definition = {
    "processor": "Processor_cond",
    "cond_inds": torch.tensor([10,11,12,13]),
    "processor_args": {"N_min": 500, "percentile2": 95},
    "processor_data": {"folder": "all_sims"},
    "flow_hyper": {"n_layers":24, "dim_notcond": 10, "dim_cond": 4, "CL":"NSF_CL2", "K": 10, "B":3, "network":"MLP", "network_args":torch.tensor([512,8,0.2])},
    "subset_params": {"cond_fn": "cond_M_stars_2age_avZ", "use_fn": "MW_like_galaxy", "train_use_fn_constructor": "construct_MW_like_galaxy_leavout", "leavout_indices": None},
    "data_prep_args": {"transformation_functions":("np.log10",), "transformation_indices":(torch.tensor([10]),), "inverse_transformations":("10**x",)}
}