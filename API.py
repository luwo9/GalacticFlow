#The purpose of this is to allow a way of working with the model, that is not as flexible as maybe the usual workflow, but is more user friendly and easier to use.
#Therfore an abstarct model class is defined that combines the pytorch model with all the processing steps that are needed to get the model to work.
#In principle it can be adjusted as desired, if things in the workflow change, recovering full flexibility.

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
import pandas as pd
import torch.multiprocessing as mp

func_handle ={
    "Processor_cond": processing.Processor_cond,
    "Processor": processing.Processor,
    "NSFlow": flowcode.NSFlow,
    "NSF_CL": flowcode.NSF_CL,
    "NSF_CL2": flowcode.NSF_CL2,
    "MLP": flowcode.MLP,
    "np.log10": np.log10,
    "10**x": lambda x: 10**x,
    "construct_MW_like_galaxy_leavout": ext.construct_MW_like_galaxy_leavout,
    "cond_M_stars_2age_avZ": ext.cond_M_stars_2age_avZ,
    "cond_M_stars": ext.cond_M_stars,
    "construct_all_galaxies_leavout": ext.construct_all_galaxies_leavout,

}

def _handle_func(func):
    if isinstance(func, str):
        return func_handle[func]
    else:
        return func

class GalacticFlow:
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
    get_conds(self, type_of_object):
        Get the parameters the model is conditional on.
    get_components(self, type_of_object):
        Get the components of the data.
    
    Attributes
    ----------
    Galaxies : list of dicts
        The Galaxies from the data after all the preprocessing steps, in standard format.
    train_loss_history : np.array
        The loss history during training.
    train_time : float
        The time it took to train the model.
    flow_architecture : str
        The architecture of the flow.
    

    Definition of a model
    ---------------------

    The definition exactly describes where to get the data from, how to process it, what the flow architecture is etc.
        This allows easy saving and loading of the model, as well as easy reproducibility.
        When saving the model additionally weights, loss history, scaling parameters etc. are saved and can later be loaded.
        This allows do easily make inference with the model, without having to know the exact definition of the model.
        See __init__ for details.



    
    """
    def __init__(self, definition, safe_mode=True):
        """
        Constructs a GalacticFlow model, as uniquely given by the definition.
        May directly load an existing model from a filepath or create a new one from a definition dictioanry.

        Parameters
        ----------

        definition: dict or str
            The definition of the model, containing all information, either as a dict or a file path to a saved existing model. See below for details.
        safe_mode: bool, (optional), default: True
            If True, the model is loaded without the use of pickle (=load with torch.load and weights_only=True).
            If False, arbitrary functions may be saved and loaded, but it comes with the usual risks of using pickle (arbitrary code execution).
        
        Definition of a model
        ---------------------

        The definition exactly describes where to get the data from, how to process it, what the flow architecture is etc.
        This allows easy saving and loading of the model, as well as easy reproducibility.
        When saving the model additionally weights, loss history, scaling parameters etc. are saved and can later be loaded.
        This allows do easily make inference with the model, without having to know the exact definition of the model.

        The following keys are needed for the standard model: (The Examples provided are the ones used by us for the first MW models)
        "processor": A processor class that is used to process the data. This is needed for the training and sampling. E.g. "Processor_cond"
        "processor_args": A dict of arguments that are needed to initialize the processor. E.g. {}
        "processor_data": A dict with the args that are needed to get the data from the processor, contains probably only 1 key, the data folder. E.g. {"folder": "all_sims"}
        "processor_clean": A dict with the args that are needed to clean the data from the processor. E.g. {"N_min":500}
        "flow_hyper": A dict with the hyperparmeters of the flow, see flowcode.NSFlow for details. E.g. {"n_layers":24, "dim_notcond": 10, "dim_cond": 4, "CL":"NSF_CL2", "K": 10, "B":3, "network":"MLP", "network_args":(512,8,0.2)}
        "subset_params": Dict for obtaining the right subset. Specify comp_use(optional), cond_fn, use_fn_constructor, as well as leavout_key and leavout_vals (list) to leave out galaxys during training.
            Function is first called with empty leavout_vals to get total data and then with specified levout_vals for training set. E.g. {"cond_fn": "cond_M_stars_2age_avZ", "use_fn_constructor": "construct_all_galaxies_leavout", "leavout_key": "id", "leavout_vals": []}
        "data_prep_args": Args to processor.Data_to_flow, includes transformation_functions, transformation_components, inverse_transformations, transformation_logdets(optional). E.g. {"transformation_functions":("np.log10",), "transformation_components":(["M_stars"],), "inverse_transformations":("10**x",)}

        The class will save the following things additionally:
        "std": The standard deviation attribute of the processor.
        "mean": The mean attribute of the processor.
        "coponent_names": The component names attribute of the processor.
        "cond_names": The conditional names attribute of the processor.
        "flow_dict": The state_dict of the flow.
        "loss_history": A list of the loss history during training.

        See the template definition for details and more up to date parameters.
        """
        #Definition is a dict of dicts that contains all the information needed to define the model or a file path to a saved (abstract) model
        #safe_mode = true means in case of a file path that it is loaded without the use of pickle (load with weights_only=True).
        #If it is intended to be loaded with safe_mode, the functions and classes must be given by a string and the corresponding function/class is used from the func_handle dict.
        #Without safe_mode arbitrary functions may be saved and loaded, but it is not safe, as it uses pickle.
        #Similarly, in case of safe_mode for every input instead of np.array use torch.tensor, such that no pickling is needed.

        #The following keys are needed for the standard model: (The Examples provided are the ones used by us for the MW model)
        #   "processor": A processor class that is used to process the data. This is needed for the training and sampling. E.g. "Processor_cond"
        #   "processor_args": A dict of arguments that are needed to initialize the processor. E.g. {}
        #   "processor_data": A dict with the args that are needed to get the data from the processor, contains probably only 1 key, the data folder. E.g. {"folder": "all_sims"}
        #   "processor_clean": A dict with the args that are needed to clean the data from the processor. E.g. {"N_min":500}
        #   "flow_hyper": A dict with the hyperparmeters of the flow, see flowcode.NSFlow for details. E.g. {"n_layers":24, "dim_notcond": 10, "dim_cond": 4, "CL":"NSF_CL2", "K": 10, "B":3, "network":"MLP", "network_args":(512,8,0.2)}
        #   "subset_params": Dict for obtaining the right subset. Specify comp_use(optional), cond_fn, use_fn_constructor, as well as leavout_key and leavout_vals (list) to leave out galaxys during training.
        #       Function is first called with empty leavout_vals to get total data and then with specified levout_vals for training set. E.g. {"cond_fn": "cond_M_stars_2age_avZ", "use_fn_constructor": "construct_all_galaxies_leavout", "leavout_key": "id", "leavout_vals": []}
        #   "data_prep_args": Args to processor.Data_to_flow, includes transformation_functions, transformation_components, inverse_transformations, transformation_logdets(optional). E.g. {"transformation_functions":("np.log10",), "transformation_components":(["M_stars"],), "inverse_transformations":("10**x",)}
        #The class will save the following things additionally:
        #   "std": The standard deviation attribute of the processor.
        #   "mean": The mean attribute of the processor.
        #   "coponent_names": The component names attribute of the processor.
        #   "cond_names": The conditional names attribute of the processor.
        #   "flow_dict": The state_dict of the flow.
        #   "loss_history": A list of the loss history during training.

        if isinstance(definition, str):
            #Load from file
            definition = torch.load(definition, map_location="cpu", weights_only=safe_mode)
        
        #Wheather the model is loaded from a self.save file/definition or from a user defined definition
        #If from self.save then flow_dict etc. are expected to be in the definition
        self.is_loaded = "was_saved" in definition and definition["was_saved"]

        #Create processor
        self.processor = _handle_func(definition["processor"])
        self.processor = self.processor(**definition["processor_args"])

        #Load processor attributes that are independent of the data i.e. always in the definition
        self.processor.trf_fn = tuple(_handle_func(fn) for fn in definition["data_prep_args"]["transformation_functions"])
        self.processor.trf_comp = definition["data_prep_args"]["transformation_components"]
        self.processor.trf_fn_inv = tuple(_handle_func(fn) for fn in definition["data_prep_args"]["inverse_transformations"])
        self.processor.trf_logdet = tuple(_handle_func(fn) for fn in definition["data_prep_args"]["transformation_logdets"]) if "transformation_logdets" in definition["data_prep_args"] else None

        #Create flow model
        flow_def = definition["flow_hyper"].copy()
        flow_def["CL"] = _handle_func(flow_def["CL"])
        flow_def["network"] = _handle_func(flow_def["network"])
        flow_def["network_args"] = (int(flow_def["network_args"][0]), int(flow_def["network_args"][1]), float(flow_def["network_args"][2]))
        self.flow = flowcode.NSFlow(**flow_def)

        #If the model is loaded from file recover  model + relavant processor attributes
        if self.is_loaded:
            for k,v in self._processor_attrs_from_prepare_load(definition).items():
                setattr(self.processor, k, v)
            
            self.flow.load_state_dict(definition["flow_dict"])


        #Some important attributes
        self.leavout_train = {"key": definition["subset_params"]["leavout_key"], "vals": definition["subset_params"]["leavout_vals"]}
        self.loss_history = definition["loss_history"].numpy() if "loss_history" in definition else []
        self.n_dim = definition["flow_hyper"]["dim_notcond"]
        self.n_cond = definition["flow_hyper"]["dim_cond"]
        #Save definition
        self.definition = definition
        self.is_prepared = False

    def prepare(self):
        
        if self.is_prepared:
            print("Warning: Model already prepared, repreparing it.")
        
        #Get the right functions
        cond_fn = _handle_func(self.definition["subset_params"]["cond_fn"])
        use_fn_constructor = _handle_func(self.definition["subset_params"]["use_fn_constructor"])
        trf_fn = self.processor.trf_fn
        trf_comp = self.processor.trf_comp
        trf_fn_inv = self.processor.trf_fn_inv
        trf_logdet = self.processor.trf_logdet
        comp_use = self.definition["subset_params"]["comp_use"] if "comp_use" in self.definition["subset_params"] else None
        
        #Load data
        Galaxies_raw = self.processor.get_data(**self.definition["processor_data"])

        #Clean data
        Galaxies_cleaned = self.processor.constraindata(Galaxies_raw, **self.definition["processor_clean"])

        #Choosing correct subset
        use_fn_view = use_fn_constructor(self.leavout_train["key"], [])
        use_fn_train = use_fn_constructor(self.leavout_train["key"], self.leavout_train["vals"])

        if comp_use is None:
            Galaxies = self.processor.choose_subset(Galaxies_cleaned, use_fn=use_fn_view, cond_fn=cond_fn)
            Train_Galaxies = self.processor.choose_subset(Galaxies_cleaned, use_fn=use_fn_train, cond_fn=cond_fn)
        else:
            Galaxies = self.processor.choose_subset(Galaxies_cleaned, use_fn=use_fn_view, cond_fn=cond_fn, comp_use=comp_use)
            Train_Galaxies = self.processor.choose_subset(Galaxies_cleaned, use_fn=use_fn_train, cond_fn=cond_fn, comp_use=comp_use)

        self.Galaxies = Galaxies

        #Prepare data for flow
        Data_flow = self.processor.Data_to_flow(self.processor.diststack(Train_Galaxies), trf_fn, trf_comp, trf_fn_inv, trf_logdet)

        self.Data_flow = Data_flow

        #Set flag
        self.is_prepared = True


    def get_conds(self, type_of_object):
        """
        Get the conditions of the given object. In the right order, as governed by the processor.

        Parameters
        ----------
        type_of_object : str
            The type of object to get the conditions of. E.g. "stars", "gas" or "dust".
        
        Returns
        -------

        condition_names : list of str
            The names of the conditions of the given object. E.g. ["M_star", "tau50"].
        """
        return self.processor.cond_names[type_of_object] + self.processor.cond_names["galaxy"]
    
    def get_components(self, type_of_object):
        """
        Get the components of the given object. In the right order, as governed by the processor.

        Parameters
        ----------
        type_of_object : str
            The type of object to get the components of. E.g. "stars", "gas" or "dust".
        
        Returns
        -------

        component_names : list of str
            The names of the components of the given object. E.g. ["x", "y", "z", "vx", "vy", "vz"].
        """
        return self.processor.component_names[type_of_object]
    

    def train(self, epochs, init_lr, batch_size, gamma, device, info=False, update_textfile=False):
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
        cond_names = self.get_conds("stars")
        flowcode.train_flow(self.flow, self.Data_flow, cond_names, epochs, lr=init_lr, batch_size=batch_size, gamma=gamma, loss_saver=loss_history, give_textfile_info=update_textfile)
        end = time.perf_counter()
        #Save loss history
        self.loss_history = np.array(loss_history +[end-start])

        self.flow.to("cpu")
        #gc + clear cache?
        if info:
            time_passed = (end-start)/60
            print(f"Training took about {int(time_passed/60)} hours and {int(time_passed%60)} minutes.")

    
    def general_sample(self, Condition:pd.DataFrame, reinsert_conditions="all" ,split_size=300000, GPUs=None):
        """
        Draws a sample from the model for a specified condition. For every condition one corresponding sample point is drawn.

        Parameters
        ----------

        Condition : pd.DataFrame
            Condition for which to draw a sample. Must have the columns as given by get_conds().
        reinsert_conditions : {"all", "local", "none"} (optional), default: "all"
            Whether to reinsert the condition into the sample returned.
            "all": Reinsert all condition values into the sample.
            "local": Reinsert only the condition values that are not galaxy properties but e.g. star properties (e.g. x if learning p(x|y)),
                (specified as pre_defined_cond in subset choosing).
            "none": Do not reinsert any condition values into the sample.
        split_size : int, (optional), default: 300000
            A technical parameter. The sample is queued in chunks of size split_size for sampling. This is done to avoid memory errors on GPUs, if the sample is too large.
        GPUs : list of ints, (optional), default: None
            List of GPUs to use for sampling in parallel. If None, use device the model is currently on. The integers in the list correspond to the GPU ids.

        Returns
        -------

        sample : pd.DataFrame
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

        sample = self.processor.sample_Conditional(self.flow, Condition, split_size=split_size, GPUs=GPUs, reinsert_condition=reinsert_conditions)

        sample = self.processor.sample_to_Data(sample)

        return sample
    
    def sample_galaxy(self, N_stars, parameters, reinsert_conditions="all",split_size=300000, GPUs=None):
        """
        Sample a galaxy with a desired number of stars for the given parameters.
        This is a userfrienldy way to sample from the model. Allows also sampling several galaxies at once, see Notes.

        Parameters
        ----------

        N_stars : int or list of ints
            Number of stars that should be sampled for this galaxy.
        parameters : pd.DataFrame or list of pd.DataFrames or dict or list of dicts
            The parameters of the galaxy(/-ies) to sample. The parameters must be given as a dict or pd.DataFrame with the keys/columns as given by get_conds() and one value each.
            If multiple galaxies are to be sampled, the parameters can be given as a list of dicts or pd.DataFrames, where each entry corresponds to one galaxy.
        reinsert_conditions : {"all", "galaxy", "local", "none"} (optional), default: "all"
            Whether to reinsert the condition into the sample returned. See general_sample for details.
            "galaxy": This will return a list of galaxy dicts, just like GalacticFlow.Galaxies (standard Data format).
                "local" conditions are reinserted and galactic ones are given at the "parameters" key of the returned dicts.
        split_size : int, (optional), default: 300000
            A technical parameter. The sample is queued in chunks of size split_size for sampling. This is done to avoid memory errors on GPUs, if the sample is too large.
        GPUs : list of ints, (optional), default: None
            List of GPUs to use for sampling in parallel. If None, use device the model is currently on. The integers in the list correspond to the GPU ids.
        
        Returns
        -------
        galaxy(/-ies) : array or list of arrays
            The sampled galaxy(/-ies). A galaxy has shape (N_stars, self.n_dim+self.n_cond), where n_dim is the dimension of the data.

        Notes
        -----
        The base functionality is to give 1 galaxy for 1 set of n_cond parameters. One could just loop over this method with the desired parameters to sample multiple galaxies.
        However, when sampling multiple galaxies directly with this method, the sample is drawn as one big sample and then split back into the individual galaxies requested.
        This is usually way more efficient for reasons of parallelization, initialization and python being slow.
        There are 4 'modes' of input:
        1. N_stars is an int and parameters is an array of shape (self.n_cond,). This is the base functionality. Will return the single galaxy requested.
        2. N_stars is a list of ints and parameters is a list of arrays of shape (self.n_cond,). Will return a list of galaxies. i-th galaxy has N_stars[i] stars and parameters[i] parameters.
        3. N_stars is an int and parameters is a list of arrays of shape (self.n_cond,). Will return a list of galaxies. All galaxies have N_stars stars and the i-th galaxy has parameters[i] parameters.
        4. N_stars is a list of ints and parameters is an array of shape (self.n_cond,). Will return a list of galaxies. All galaxies have parameters parameters and the i-th galaxy has N_stars[i] stars.


        Examples
        --------

        >>> use_gpus = [0,1,2,3]
        #Model is conditional in M_star and tau50
        >>> model.n_cond, model.get_conds("stars")
        2, ["M_star", "tau50"]
        #Model is learnt on 10D data space (x,y,z,vx,...)
        >>> model.n_dim, model.get_components("stars")
        10, ["x", "y", "z", "vx", "vy", "vz", "Z", "feh", "ofe", "age"]
        #Sample a single galaxy with total stellar mass of 10^10 M_sun and tau50 of 6 Gyr
        >>> parameters = pd.DataFrame({"M_star": [10^10], "tau50": [6]})
        #Sample 10^6 stars for this galaxy
        >>> galaxy = model.sample_galaxy(10^6, parameters, GPUs=use_gpus)
        >>> galaxy.shape
        (10^6, 12)
        >>> galaxy[:2]
         x   y   z    vx   vy   vz   Z   feh  ofe  age  M_star  tau50
         0.2 0.5 1.0  0.0  2.5  0.0  0.0  1   -1    9    10^10   6
         1.0 2.5 1.0  0.5  2.5  1.0  0.0  1   -1    9    10^10   6
        #Sample 3 galaxies 1: 10^10 M_sun, 6 Gyr, 2: 10^11 M_sun, 6 Gyr, 3: 10^10 M_sun, 8 Gyr
        >>> parameters = np.split(pd.DataFrame({"M_star": [10^10,10^11,10^10], "tau50": [6,6,8]}), 3) #Note: This method also works without the np.split
        #Sample 10^6 stars for each galaxy
        >>> galaxy1, galaxy2, galaxy3 = model.sample_galaxy(10^6, parameters, GPUs=use_gpus)
        #Saple with different number of stars for each galaxy
        >>> galaxy1, galaxy2, galaxy3 = model.sample_galaxy([10^6,10^7,10^6], parameters, GPUs=use_gpus)
        """
        if reinsert_conditions == "galaxy":
            reinsert_conditions = "local"
            return_galaxy = True
        else:
            return_galaxy = False

        #Get all parameters in the right shape
        #Check if parameters are 1 array or list of arrays, be error tolerant
        #If one time this will allow empty arrays (unconditional sampling), just check if len(parameters) == 0
        multiple_params_given = len(np.array(parameters).shape) > 2
        multiple_N_stars_given = not isinstance(N_stars, int)

        #Convert dict or list of dicts to pd.DataFrame or list of pd.DataFrames
        #And save Column names
        if isinstance(parameters, (dict, pd.DataFrame)):
            parameters = pd.DataFrame(parameters, index=[0])
            Column_names = parameters.columns.to_list()
        elif isinstance(parameters, list):
            parameters = [pd.DataFrame(par, index=[0]) for par in parameters]
            Column_names = parameters[0].columns.to_list()
        elif isinstance(parameters, pd.DataFrame) and len(parameters) > 1:
            Column_names = parameters.columns.to_list()
            parameters = np.split(parameters, len(parameters))
        else:
            raise TypeError("Parameters must be a dict, a list of dicts or a pd.DataFrame.")

        #Convert to np.array
        #Column names are already saved
        par_sample = np.array(parameters)
        par_sample = par_sample[:,0] if multiple_params_given else par_sample
        N_stars = np.array(N_stars)


        
        if not multiple_params_given and multiple_N_stars_given:
            #If only one set of parameters is given, but multiple N_stars, interpret as same galaxy with different number of stars (even if that makes not as much sense)
            N_sample = np.sum(N_stars)
        else:
            N_sample = N_stars

        par_sample = np.repeat(par_sample, N_sample, axis=0)

        par_sample = pd.DataFrame(par_sample, columns=Column_names)

        sample = self.general_sample(par_sample, split_size=split_size, GPUs=GPUs, reinsert_conditions=reinsert_conditions)

        #Now split the sample into the individual galaxies
        if multiple_params_given and not multiple_N_stars_given:
            #If multiple sets of parameters are given, but only one N_stars, interpret as different galaxies with same number of stars
            N_split = np.repeat(N_stars, len(parameters))
        else:
            N_split = N_stars
        
        galaxies = self.processor.galaxysplit(sample, N_split)

        if return_galaxy:
            parameters = [parameters] if not isinstance(parameters, list) else parameters
            galaxies = [{"stars":df, "parameters":par} for df, par in zip(galaxies, parameters)]

        if not multiple_params_given and not multiple_N_stars_given:
            #If only one set of parameters and one N_stars is given, return only one galaxy
            galaxies = galaxies[0]

        return galaxies
    
    def general_pdf(self, X, split_size=300000, GPUs=None):
        """
        Evaluate the log probability density function at a given set of points.

        Parameters
        ----------
        X : pd.DataFrame
            Points on which to evaluate the pdf. Must contain columns for all components (get_components()) and conditions (extra elements in get_conds()).
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
        >>> model.n_dim, model.get_components("stars")
        10, ["x", "y", "z", "vx", "vy", "vz", "Z", "feh", "ofe", "age"]
        #Assume the flow was learned conditional also in the x coordinate, not just in galactic parameters
        >>> model.n_cond, model.get_conds("stars")
        4, ["x", M_star", "tau50", "avZ"]
        >>> Conditions_parameter = pd.DataFrame({"M_star": [10^10,10^11,10^10], "tau50": [6,6,8], "avZ": [0.1,0.2,0.1]})# Note that this does not contain x !
        >>> Data = pd.DataFrame(np.random.rand(4,10), columns=model.get_components("stars")) #Note that this already contains x !
        >>> X = pd.concat([Data, Conditions_parameter], axis=1)
        >>> pdf = model.general_pdf(X, GPUs=use_gpus)
        >>> pdf.shape
        (4,)
        >>> pdf
        array([-1.2,-1.5,1.8,-1.9])
        """

        self.flow.eval()
        pdf = self.processor.log_prob(self.flow, X, split_size=split_size, GPUs=GPUs)

        return pdf
    
    def pdf_galaxy(self, galaxy, parameters, split_size=300000, GPUs=None):
        """
        Evaluate the log probability density function for a galaxy with a given set of parameters.
        This is a userfrienldy way to evaluate the pdf.

        Parameters
        ----------

        galaxy : pd.DataFrame
            The points on which to evaluate the pdf. Must contain columns for all components (get_components()).
        parameters : pd.DataFrame or dict
            The parameters of the galaxy to evaluate the pdf for. Must be given as a dict or pd.DataFrame with the keys/columns as given by get_conds() and one value each.
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

        #Get all parameters in the right shape
        if isinstance(parameters, dict):
            parameters = pd.DataFrame(parameters, index=[0])
        
        #Vertically stack parameters
        columns = parameters.columns.to_list()
        parameters = parameters.values
        parameters = np.repeat(parameters, len(galaxy), axis=0)
        parameters = pd.DataFrame(parameters, columns=columns)

        #Horizontally stack galaxy and parameters
        X = pd.concat([galaxy, parameters], axis=1)

        pdf = self.general_pdf(X, split_size=split_size, GPUs=GPUs)

        return pdf
    
    def save(self, path, ensure_trained=True):
        """
        Save this model to a file. This will not just save the pytorch model, but the whole state of the GalacticFlow model/object see Notes.
        This file will later be used to load again at initialization.

        Parameters
        ----------

        path : str or None
            The path to save the model to. If None, the model will not be saved to a file, but only returned.
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
        untrained = len(self.loss_history) == 0
        
        unprepared = not hasattr(self.processor, "mu") or self.processor.mu is None

        if ensure_trained:
            if unprepared:
                raise RuntimeError("Model has never been prepared. Cannot save. If you want to save anyway, set ensure_trained=False.")
            elif untrained:
                raise RuntimeError("Model has never been trained. Cannot save. If you want to save anyway, set ensure_trained=False.")
            else:
                #Everything is okay
                save_processor = self._processor_attrs_from_prepare_to_save()
        else:
            if unprepared:
                print("Warning: Model has never been prepared. Saving anyway with mu=None and std=None.")
                save_processor = self._processor_attrs_from_prepare_to_save(prepared=False)
                #Setting to None will ensure that it is saved, but can not be used for inference by accident
                #But this may not be catched in inference if a custom Processor Class does ,e.g., define mu and std as 0 and 1 in __init__... maybe some better way to do this?
            elif untrained:
                print("Warning: Model has never been trained. Saving anyway.")
                save_processor = self._processor_attrs_from_prepare_to_save()
            else:
                #Everything is okay
                save_processor = self._processor_attrs_from_prepare_to_save()

        #Use definition of model to save + extra keywords (see __init__)
        save_dict = self.definition.copy()

        #Add processor state
        save_dict.update(save_processor)
        #watch flow's device? but loading does already so maybe not that bad
        save_dict["flow_dict"] = self.flow.state_dict()
        save_dict["loss_history"] = torch.tensor(self.loss_history)


        save_dict["was_saved"] = True #Flag that the model has been saved
        if path is None:
            return save_dict
        else:
            torch.save(save_dict, path)

    def _processor_attrs_from_prepare_to_save(self, prepared=True):
        if prepared:
            mu_vals = torch.from_numpy(self.processor.mu.values)
            std_vals = torch.from_numpy(self.processor.std.values)
            mu_names = self.processor.mu.index.to_list()
            std_names = self.processor.std.index.to_list()
            return_dict = {"mu_vals": mu_vals, "std_vals": std_vals, "mu_names": mu_names, "std_names": std_names,
                           "component_names": self.processor.component_names, "cond_names": self.processor.cond_names}
        else:
            return_dict = {"mu": None, "std": None,
                           "component_names": self.processor.component_names, "cond_names": self.processor.cond_names}
            
        return return_dict
    
    def _processor_attrs_from_prepare_load(self, save_dict):
        if save_dict["mu_vals"] is not None and save_dict["std_vals"] is not None:
            save_dict["mu"] = pd.Series(save_dict["mu_vals"].numpy(), index=save_dict["mu_names"])
            save_dict["std"] = pd.Series(save_dict["std_vals"].numpy(), index=save_dict["std_names"])
            return_dict = {"mu": save_dict["mu"], "std": save_dict["std"],
                           "component_names": save_dict["component_names"], "cond_names": save_dict["cond_names"]}
        else:
            return_dict = {"mu": None, "std": None,
                           "component_names": save_dict["component_names"], "cond_names": save_dict["cond_names"]}
        
        return return_dict
    
    #Some important attributes that should be accessible
    @property
    def train_loss_history(self):
        """
        The loss history of the model during training.
        """
        return self.loss_history[:-1]
    
    @property
    def train_time(self):
        """
        The time it took to train the model in seconds.
        """
        return self.loss_history[-1]

    @property
    def flow_architecture(self):
        """
        The architecture of the normalizing flow.
        """
        flow_hypers = self.definition["flow_hyper"]

        #Build architecture string
        printout_string = f"""Data dim: {self.n_dim}, Condition dim: {self.n_cond}

Flow architecture:
Type of coupling layer: {flow_hypers["CL"]}{f", split fraction of dimensions{flow_hypers['split']}" if flow_hypers["CL"]=="NSF_CL" else ""}
Number of layers: {flow_hypers["n_layers"]}
Number of spline bins: {flow_hypers["K"]}
Spline range: {flow_hypers["B"]}
Base network: {flow_hypers["network"]}

Base network architecture:
Number of layers: {int(flow_hypers["network_args"][1])}
Number of neurons per layer: {int(flow_hypers["network_args"][0])}
Leaky ReLU slope: {flow_hypers["network_args"][2]}"""

        #Now format the string such that is is printed nicely in a jupyter notebook when using display()
        printout_string = printout_string.replace("\n", "<br>")
        printout_string = printout_string.replace(" ", "&nbsp;")


        return printout_string

#Example for a definition dictionary
#We recommend to use this as a template for your own models. Simply copy and paste and change the values accordingly.
example_definition = {
    "processor": "Processor_cond",
    "processor_args": {},
    "processor_data": {"folder": "all_sims"},
    "processor_clean": {"N_min":500},
    "flow_hyper": {"n_layers":24, "dim_notcond": 10, "dim_cond": 4, "CL":"NSF_CL2", "K": 10, "B":3, "network":"MLP", "network_args":torch.tensor([512,8,0.2])},
    "subset_params": {"cond_fn": "cond_M_stars_2age_avZ", "use_fn_constructor": "construct_all_galaxies_leavout", "leavout_key": "id", "leavout_vals": []},
    "data_prep_args": {"transformation_functions":("np.log10",), "transformation_components":(["M_stars"],), "inverse_transformations":("10**x",)}
}

#Function for seperate process, parallel training

def _train_seperate(definition, gpu_id, unique_id, queue, loading_result_finished, **train_kwargs):
    """
    Target function to be run in a seperate process for parallel training.
    """
    #Create model
    model = GalacticFlow(definition)

    #Load data
    model.prepare()

    train_kwargs["device"] = f"cuda:{gpu_id}" if gpu_id is not None else "cpu"

    try:
        #Train
        model.train(**train_kwargs)
        #Save result
        save_dict = model.save(None)

    except ValueError as e:
        #If training fails, return error
        save_dict = e

    #Put result in queue
    queue.put((unique_id, gpu_id, save_dict))
    #Wait for loading to finish
    loading_result_finished.wait()


    # #Test mode...
    # try:
    #     time.sleep(5)
    #     random_int = np.random.randint(0,100)
    #     if random_int > 20:
    #         print(f"Call with rank {unique_id}, gpu_id {gpu_id} and {train_kwargs} is supposed to crash.")
    #         distr = torch.distributions.Normal(torch.zeros(2), torch.ones(2))
    #         point = torch.ones(2)*torch.nan
    #         result = distr.log_prob(point)
    #     else:
    #         print(f"Call with rank {unique_id}, gpu_id {gpu_id} and {train_kwargs} is supposed to work.")
    #         result = definition["identif"]+torch.randn(10**6)/100
    # except ValueError as e:
    #     result = e
    # queue.put((unique_id, gpu_id, result))

    # #time.sleep(1)
    # #Wait for loading to finish
    # #We do this part below such that the the child process only terminates after the tensor is fully obtained
    # #Not just the reference to it, which is sent over the queue
    # loading_result_finished.wait()


def train_GF(models, GPUs, train_kwargs, filenames=None, max_restart = 2):
    """
    Train a list of GalacticFlow models in parallel on a list of GPUs.
    Each model is trained in a seperate process on its own GPU. The processes are monitored and restarted if they crash.
    If training is done the GPU is freed again.

    Parameters
    ----------

    models : GalacticFlow or dict or list of GalacticFlow or dict
        The models to train. Can be a single model or a list of models. If a list is given, the models are trained in parallel.
        If a dict is given it must be a definition dictionary (see GalacticFlow) and the corresponding model is trained.
    GPUs : list of ints or None
        The GPUs to use for training. The integers in the list correspond to the GPU ids.
        If None, train all models on CPU. Single elemtens in the list can be None, then the corresponding model is trained on CPU.
        This may be faster if you have more models then GPUs.
    train_kwargs : dict or list of dicts
        The keyword arguments to pass to the train method of the models. If a list is given, the corresponding arguments are passed to the corresponding model.
        If multiple models are given but only one set of arguments, the same arguments are passed to all models.
        See GalacticFlow.train() for details. The "device" argument does not need to be given, it is set automatically as described in GPUs.
    filenames : str or list of strs or None (optional), default: None
        The filenames to save the models to. If None, the models are not saved to a file.
        If a list is given, the corresponding models are saved to the corresponding files.
    max_restart : int (optional), default: 2
        The maximum number of times a model is restarted if it crashes. If a model crashes more often, it is not restarted again and the training is considered failed.
        Failed models will not be saved to a file and the corresponding entry in models_out is None.

    Returns
    -------

    models_out : dict or list of dicts or None
        The trained models. If filenames is None, the models are not saved to a file and the models are returned as dicts.
        Otherwise the models are saved to the corresponding files and None is returned.
        The order of the models is the same as the order of the input models.
        Note that if a model crashes more often than max_restart, the corresponding entry in models_out is None.


    Examples
    --------

    >>> use_gpus = [0,1,2]
    >>> model_def1 = {...}
    >>> model_def2 = {...}
    >>> model1 = GalacticFlow({...})
    >>> models = [model1, model_def1, model_def2]
    >>> filenames = ["model1.pth", "model2.pth", "model3.pth"]
    >>> train_kwargs = [{"epochs": 10, "batch_size": 1024, "init_lr": 0.00009, "gamma": 0.998, "update_textfile":f"model{i}.txt"} for i in range(3)]
    >>> train_GF(models, use_gpus, train_kwargs, filenames)
    >>> #Now model1, model2 and model3 are trained on GPUs 0,1,2,3 respectively and saved to model1.pth, model2.pth and model3.pth
    """

    #Check the models input
    if isinstance(models, (GalacticFlow, dict)):
        models = [models]
        
        if not isinstance(filenames, list):
            filenames = [filenames]
        if not isinstance(train_kwargs, list):
            train_kwargs = [train_kwargs]
    elif isinstance(models, list):
        #Convert to list of dicts
        models = [model if isinstance(model, dict) else model.save(None) for model in models]
        if filenames is None:
            filenames = [None]*len(models)
        elif not isinstance(filenames, list):
            raise ValueError("If models is a list, filenames must be a list of the same length.")
        elif len(filenames) != len(models):
            raise ValueError("If models is a list, filenames must be a list of the same length.")
        if not isinstance(train_kwargs, list):
            train_kwargs = [train_kwargs]*len(models)
    if GPUs is None:
        GPUs = [None]*len(models)
    if not isinstance(GPUs, list):
            GPUs = [GPUs]

    #Give every model a unique id
    #Keep track of:
    # - which models are done
    # - which models are still to do
    #The results (in the same order as the input) are stored in models_out
    # - The number of restarts after a crash
    # - The corresponding signals to wait for the loading to finish
    indices_to_do = list(range(len(models)))
    indices_done = []
    models_out = [None]*len(models)
    restarts = [0]*len(models)
    signals = [None]*len(models)

    #Prepare multiprocessing
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    #Start processes; fill all GPUs a first time
    processes = []
    for gpu_id in GPUs:
        for unique_id in range(len(models)):
            if unique_id in indices_to_do:
                #print(f"Starting training of model {unique_id} on GPU {gpu_id}.")
                event = ctx.Event()
                signals[unique_id] = event
                p = ctx.Process(target=_train_seperate, args=(models[unique_id], gpu_id, unique_id, queue, event), kwargs=train_kwargs[unique_id])
                p.start()
                processes.append(p)
                indices_to_do.remove(unique_id)
                break
        
    #Monitor processes, restart crashed ones and refill emtpy slots/gpus
    while len(indices_done) < len(models):
        unique_id, gpu_id, save_dict = queue.get()
        #We do this part below such that the the child process only terminates after the tensor is fully obtained
        #Not just the reference to it, which is sent over the queue
        loading_result_finished = signals[unique_id]
        loading_result_finished.set()
        if isinstance(save_dict, ValueError) and restarts[unique_id] < max_restart:
            #print(f"Restarting training of model {unique_id} on GPU {gpu_id}.")
            event = ctx.Event()
            signals[unique_id] = event
            p = ctx.Process(target=_train_seperate, args=(models[unique_id], gpu_id, unique_id, queue, event), kwargs=train_kwargs[unique_id])
            p.start()
            processes.append(p)
            restarts[unique_id] += 1
        else:
            #Model finished successfully or crashed too often, restart a new one
            #print(f"Finished training of model {unique_id} on GPU {gpu_id}.")
            indices_done.append(unique_id)

            #Save model
            if isinstance(save_dict, ValueError):
                save_dict = None
            models_out[unique_id] = save_dict
            #Also save to file, such that is is already available
            if filenames[unique_id] is not None and save_dict is not None:
                #print(f"Saving model {unique_id} to {filenames[unique_id]}.")
                torch.save(save_dict, filenames[unique_id])

            #Restart a new process if there are still models to train
            for unique_id in range(len(models)):
                if unique_id in indices_to_do:
                    #print(f"Starting training of model {unique_id} on GPU {gpu_id}.")
                    event = ctx.Event()
                    signals[unique_id] = event
                    p = ctx.Process(target=_train_seperate, args=(models[unique_id], gpu_id, unique_id, queue, event), kwargs=train_kwargs[unique_id])
                    p.start()
                    processes.append(p)
                    indices_to_do.remove(unique_id)
                    break

    #Wait for all processes to finish
    for p in processes:
        p.join()

    if filenames[0] is None:
        return models_out

    
