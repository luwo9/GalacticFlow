import numpy as np
import pandas as pd
#Externalized functions for condition selection and galaxy selection here to make the mai file, where this is normally expected to be, more readable.
#And also to give examples of how this works.
def _MW_like_galaxy(galaxy):
    """
    Selects galaxies that are Milky Way like.
    Namely the galaxies have to have stellar mass plus dark matter mass greater than $5\cdot10^11$.

    Parameters
    ----------

    galaxy : dict
        Dict containing the data of one galaxy. See Processor_cond for more information.

    Returns
    -------

    bool
        True if the galaxy is considered Milky way like, False otherwise.
    """
    M_star = galaxy["galaxy"]["M_stars"]
    M_dm_g = galaxy["galaxy"]["M_dm"]
    return M_star+M_dm_g > 5e11


#Function simmilar to MW_like_galaxy, but forbids galaxies with a certain dm mass. (Leave one/multiple out)
def construct_MW_like_galaxy_leavout(key, forbidden_values):
    """
    Constructs a function that selects galaxies that are Milky Way like, but forbids galaxies with certain values of a certain key in the galaxy["galaxy"] dict.
    Intended to un-select galaxies with a certain property, to leave them out of the training.
    Examples would be to leave out galaxies with certain unique id(s), or certain dark matter mass(es).

    Parameters
    ----------

    key : str
        Key of the galaxy["galaxy"] dict, that should be used to determine if a galaxy is forbidden.
    forbidden_values : array or list
        Values of the key, that shoulbe be excluded.


    Returns
    -------

    function
        Function that selects galaxies that are Milky Way like, but forbids galaxies with certain values of a certain key in the galaxy["galaxy"] dict.
    """
    #print("Remember it's best to once choose a subset to view, use those DM masses and choose a 2nd subset for training. SORT Mdm before choosing indices")
    def MW_like_galaxy_leavout(galaxy):
        """
        Selects galaxies that are Milky Way like, but forbids galaxies with certain values of a certain key in the galaxy["galaxy"] dict.
        For the use as use_fn in choose_subset.
        Wheater a galaxy is Milky Way like is determined by MW_like_galaxy.

        Parameters
        ----------

        galaxy : dict
            Dict containing the data of one galaxy.

        Returns
        -------

        bool
            True if the galaxy is considered Milky way like and has none of forbidden values, False otherwise.
        """
        #Test if is MW like
        is_MW_like = _MW_like_galaxy(galaxy)
        #Test if has forbidden dm mass
        galaxy_value = galaxy["galaxy"][key]
        is_included = np.isin(galaxy_value, forbidden_values, invert=True).item()

        return is_MW_like and is_included
    
    return MW_like_galaxy_leavout




def construct_all_galaxies_leavout(key, forbidden_values):
    """
    Constructs a function that selects all galaxies, but forbids certain values of a certain key in the galaxy["galaxy"] dict.
    See construct_MW_like_galaxy_leavout for more information.
    """
    def all_galaxies_leavout(galaxy):
        """
        Selects all galaxies, but forbids certain, predetermined, galaxies.
        See MW_like_galaxy_leavout for more information.
        """
        #Test if has forbidden dm mass
        galaxy_value = galaxy["galaxy"][key]
        is_included = np.isin(galaxy_value, forbidden_values, invert=True).item()

        return is_included
    
    return all_galaxies_leavout


#Condition functions, use intended as cond_fn in choose_subset

def cond_M_stars(galaxy):
    """
    Simply returns the stellar mass as condition. For the use as cond_fn in choose_subset.

    Parameters
    ----------

    galaxy : dict
        Dict containing the data of one galaxy.

    Returns
    -------

    Condition : pd.DataFrame
        Conditions of the galaxy, in this case the stellar mass.
    """
    M_star = galaxy["galaxy"]["M_stars"]
    return pd.DataFrame({"M_stars": [M_star]})

def cond_M_stars_2age_avZ(galaxy):
    """
    Returns a condition consisting of the 4 values stellar mass, median age, 10th percentile age and mean metallicity.
    For the full documentation of such a function see cond_M_stars.
    """
    stars = galaxy["stars"]
    tau50 = np.median(stars["age"])
    tau10 = np.percentile(stars["age"], 10)
    Z_av = np.mean(stars["Z"])
    M_star = galaxy["galaxy"]["M_stars"]
    return pd.DataFrame({"M_stars": [M_star], "tau50": [tau50], "tau10": [tau10], "Z_av": [Z_av]})


#Functions used for data preperation in Data_to_flow

def logdet_log10(x:pd.DataFrame, _):
    """
    Calculates the logartihm of the determinant of the Jacobian of the transformation x -> log10(x).
    It is given by $1/\ln(10) \sum_i 1/x_i$. The sum is taken along the dimension of x.

    Parameters
    ----------
    x : pd.DataFrame
        Data to be transformed.

    Returns
    -------

    array
        Array with the logartihm of the determinant of the Jacobian of the transformation x -> log10(x).
    """
    x = x.values
    jacobian = 1/np.log(10)*1/x
    log_det_jacobian = np.log(jacobian).sum(axis=1)
    return log_det_jacobian


#Old
def tanh_smoothing(x):
    """
    Applies a tanh smoothing to the data, following https://arxiv.org/pdf/2205.01129.pdf. This avoids the flow to be trained on sharp edges.
    Applies the function f(x) = sign(x)*arctanh(|x|) elementwise.

    Parameters
    ----------

    x : array
        Array with values, that should be smoothed.
    
    Returns
    -------

    array
        Array with smoothed values.
    """
    return np.sign(x)*np.arctanh(np.abs(x))

def tanh_smoothing_inv(y):
    """
    Inverse of tanh_smoothing, following https://arxiv.org/pdf/2205.01129.pdf.
    Applies the function f(x) = sign(x)*tanh(|x|) elementwise.

    Parameters
    ----------

    y : array
        Array with values following tanh_smoothing.
    
    Returns
    -------

    array
       Physical (unsmoothed) values of y. (Inverse of tanh_smoothing)
    """
    return np.sign(y)*np.tanh(np.abs(y))

def tanh_smoothing_logdet(x):
    """
    Jacobian of tanh_smoothing, used to transform the coresponding pdf.
    Uses the derivative $f(x) = 1/(1-x^2)$, and sums the logartithm along the dimension of x.

    Parameters
    ----------

    x : array
        Array with unsmoothed values.

    Returns
    -------

    array
        Array with the derivative of tanh_smoothing at the values of x.

    Notes
    -----

    Does not follow https://arxiv.org/pdf/2205.01129.pdf. This is because in the paper r (the spherical radius) is smoothed resulting in different derivatives.
    """
    return np.log(1/(1-x**2)).sum(axis=1)

def atanh_with_linear_tail(x):
    """
    Aplies a 1 sided atanh smoothing with linear tail to the data, simmilar to https://arxiv.org/pdf/2205.01129.pdf.
    This smoothes out sharp edges, which can be challenging for the flow to learn.
    Aplies elementwise the function f(x) = arctanh(x) if x >= 0, and f(x) = x if x < 0.

    Parameters
    ----------

    x : array
        Array with values, that should be smoothed.
    
    Returns
    -------

    array
        Array with smoothed values.
    """
    y = x.copy()
    y[y>=0] = np.arctanh(y[y>=0])
    return y

def atanh_with_linear_tail_inv(y):
    """
    Inverse of atanh_with_linear_tail.
    Aplies elementwise the function f(x) = tanh(x) if x >= 0, and f(x) = x if x < 0.

    Parameters
    ----------

    y : array
        Array with values following atanh_with_linear_tail, to be unsmoothed.
    
    Returns
    -------

    array
         Physical (unsmoothed) values of y. (Inverse of atanh_with_linear_tail)
    """
    x = y.copy()
    x[x>=0] = np.tanh(x[x>=0])
    return x

def atanh_with_linear_tail_logdet(x):
    ...

class tanh_smooth():
    """
    Class for tanh smoothing of data. This is used to avoid the flow to be trained on sharp edges.
    This class is made for one-side smoothing, using a atanh function with linear tail, at the maximum or minimum of the data.
    Intended to be used with the Data_to_flow method of Processor_Cond class. However for inverting the smoothing, values calculated in the smoothing function are needed.
    To avoid returning these values and constructing a fitting inverse function to be used as callable input to Data_to_flow, this class is used.
    An instance of this class is created and its methods are used as callable input to Data_to_flow,
    while the object assures all needed values are stored in the background and automatically accessed in the inverse function.
    Thus, the intended use is as follows:
    1. Create an instance of this class, with the desired kind of smoothing (max or min).
    2. Use object.smooth as callable (as input to Data_to_flow).
    3. Use object.inv_smooth as callable (as input to Data_to_flow).

    Parameters
    ----------

    kind : {"max", "min"}
        Kind of smoothing, either at the maximum or minimum of the data.
    c : float, default=1.000001
        Constant do describe the smoothing behaviour. Larger c means larger spread of the smoothing.
        The atanh function diverges at x=1 and x=-1, thus the maximum is scaled to +-1/c instead of +-1.
    
    Notes
    -----
    The function scales values accordingly relative to the maximum or minimum of the data, to be properly transformed by the atanh function.
    For maximum/minimum at 0, the scaling is done relative to the mean of the data.
    The scaling function includes a saftey mechanism to avoid values to close to 0, but not 0.
    E.g. if a quantity has a minimum at not exactly zero, but very close, while the typical value is much larger, the scaling relative to minimum would lead to large values,
    that are far from 1 or -1, which leads to them not being affected by the atanh function. In this case the scaling is also done relative to the mean.
    """
    def __init__(self, kind, c=1.000001):
        self.c = c
        self.kind = kind
    
    def _scale(self, x):
        if self.kind == "max":
            self.max = np.max(x, axis=0)
            self.sign = np.sign(self.max)
            self.max = np.abs(self.max)

            #Also include a saftey net if the max is to close to 0 in comparison to the mean.
            self.mean = np.abs(np.mean(x, axis=0))
            y = np.where(self.max/self.mean < 1e-2, ((x - self.sign*self.max)/self.mean + 1)/self.c, 0)

            #If max is 0 set p_z_e to 1 to avoid division by 0 the value of y is not important as it will be overwritten in this case.
            p_z_e = np.where(self.max == 0, 1, self.max)

            y = np.where(self.sign > 0 & ~(self.max/self.mean < 1e-2), (x/p_z_e)/self.c, y)
            y = np.where(self.sign < 0 & ~(self.max/self.mean < 1e-2), (x/p_z_e +2)/self.c, y)
        
        elif self.kind == "min":
            self.min = np.min(x, axis=0)
            self.sign = np.sign(self.min)
            self.min = np.abs(self.min)

            self.mean = np.abs(np.mean(x, axis=0))
            y = np.where(self.min/self.mean < 1e-2, ((x - self.sign*self.min)/self.mean - 1)/self.c, 0)
            p_z_e = np.where(self.min == 0, 1, self.min)
            y = np.where(self.sign > 0 & ~(self.min/self.mean < 1e-2), (x/p_z_e -2)/self.c, y)
            y = np.where(self.sign < 0 & ~(self.min/self.mean < 1e-2), (x/p_z_e)/self.c, y)

            
        return y

    
    def _scale_inv(self, y):
        if self.kind == "max":
            x = np.where(self.max/self.mean < 1e-2, (y*self.c-1)*self.mean+self.sign*self.max, 0)
            x = np.where(self.sign > 0 & ~(self.max/self.mean < 1e-2), y*self.c*self.max, x)
            x = np.where(self.sign < 0 & ~(self.max/self.mean < 1e-2), (y*self.c-2)*self.max, x)
        
        elif self.kind == "min":
            x = np.where(self.min/self.mean < 1e-2, (y*self.c+1)*self.mean+self.sign*self.min, 0)
            x = np.where(self.sign > 0 & ~(self.min/self.mean < 1e-2), (y*self.c+2)*self.min, x)
            x = np.where(self.sign < 0 & ~(self.min/self.mean < 1e-2), y*self.c*self.min, x)

        return x

        
    def smooth(self, x):
        y = self._scale(x)
        if self.kind == "max":
            y = atanh_with_linear_tail(y)
        elif self.kind == "min":
            y = -atanh_with_linear_tail(-y)
        
        return y
    
    def smooth_inv(self, y):
        if self.kind == "max":
            x = atanh_with_linear_tail_inv(y)
        elif self.kind == "min":
            x = -atanh_with_linear_tail_inv(-y)
        
        x = self._scale_inv(x)

        return x

