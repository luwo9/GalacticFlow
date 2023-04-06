import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

standard_zoomout = 1.2
comp_names = "xyz"


def residual_combined(C):
    """
    Function calculating the residuals for use a binned statistic plot.
    Reiduals are defined as the difference between the mean of the data and the mean of the flow, divided by the mean of the data.
    Assumes complex input. If the imaginary part is 0, it is assumed to be data, otherwise it is assumed to be flow.

    Parameters
    ----------

    C : np.ndarray, dtype=complex
        The data and flow samples for each bin to calculate the residuals for. Shape (N,)

    Returns
    -------

    residual : float
        The residual for the bin, to be used in the plot.
    """
    C_ = np.array(C)
    is_data = np.imag(C_) == 0
    C_data = np.real(C_[is_data])
    C_flow = np.real(C_[~is_data])
    if len(C_data)==0 or len(C_flow)==0:
        return np.nan
    mean_data = np.mean(C_data)
    mean_flow = np.mean(C_flow)
    return (mean_data-mean_flow)/mean_data


def get_result_plots(data_true_, data_flow_, label="", format_="png", dpi=300):
    """
    Plot the results of the flow for a single galaxy. Makes 4 plots:
    1. Corner plot of the data and the flow in the x,y,z plane
    2. Binned hexagonal plot in the x,y plane for  average [Fe/H], [O/Fe] and age, plots data. flow and residuals.
    3. Histograms for every component, comparing data and flow
    4. Corner plot of all components, comparing data and flow

    Parameters
    ----------

    data_true_ : np.ndarray
        The (true) galaxy data, shape (N, 10)
    data_flow_ : np.ndarray
        The flow sample for the galaxy, shape (N, 10)
    label : str, optional, default: ""
        Label to add to the plot titles, when saving
    format_ : str, optional, default: "png"
        Format to save the plots in
    dpi : int, optional, default: 300
        DPI resolution to save the plots in    
    """
    
    data_true = data_true_.T
    data_flow = data_flow_.T
    names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'metals', '[Fe/H]', '[O/Fe]', 'age/Gyr']
    
    
    #2D Plots xyz corner
    ind_array = np.array([[0,1],[0,2],[1,2]])
    ax_ind_array = np.array([[0,0],[1,0],[1,1]])

    vmax = -100
    vmin = 1
    for ind1, ind2 in ind_array:
        res = plt.hexbin(data_true[ind1], data_true[ind2], gridsize=150, cmap="magma", bins="log")
        plt.close()
        bin_results = res.get_array().data
        vmax = np.maximum(bin_results.max(), vmax)
        #vmin = np.minimum(bin_results.min(), vmin)



    fig1, axs1 = plt.subplots(2,4, sharex = "all", sharey = "all", figsize=(16,8), layout="compressed")
    #Data first
    lim_max = 10**-6
    for (ind1, ind2), (ax_ind1, ax_ind2) in zip(ind_array, ax_ind_array):
        ax = axs1[ax_ind1][ax_ind2]
        im1 = ax.hexbin(data_true[ind1], data_true[ind2], gridsize=150, cmap="magma", bins="log", vmin=vmin, vmax=vmax)
        ax.set_xlabel(comp_names[ind1] if ax_ind1 == 1 else "")
        ax.set_ylabel(comp_names[ind2] if ax_ind2 == 0 else "")
        ax.text(0.02, 0.98, "Data", ha="left", va="top", transform=ax.transAxes, color="white")
        lim_max = np.maximum(np.max(np.abs(ax.get_xlim()+ax.get_ylim())), lim_max)
    
    lim_max *= standard_zoomout

    ax_ind_array[:,1] += 2
    for (ind1, ind2), (ax_ind1, ax_ind2) in zip(ind_array, ax_ind_array):
        include = (data_flow[ind1]<=lim_max)&(data_flow[ind2]<=lim_max)
        ax = axs1[ax_ind1][ax_ind2]
        im1 = ax.hexbin(data_flow[ind1,include], data_flow[ind2,include], gridsize=int(150*standard_zoomout), cmap="magma", bins="log", vmin=vmin, vmax=vmax)
        ax.set_xlabel(comp_names[ind1] if ax_ind1 == 1 else "")
        ax.set_ylabel(comp_names[ind2] if ax_ind2 == 0 else "")
        ax.text(0.02, 0.98, "Flow", ha="left", va="top", transform=ax.transAxes, color="white")


    for ax in axs1.ravel():
        ax.set_xlim(-lim_max,lim_max)
        ax.set_ylim(-lim_max,lim_max)
        ax.set_box_aspect(1)
        ax.set_facecolor(matplotlib.colormaps["magma"](0))

    fig1.suptitle("<N> corner plot. Left: data, right: sample")
    plt.colorbar(im1, ax=axs1, pad=0.03, aspect=33, shrink=1)

    plt.delaxes(axs1[0][1])
    plt.delaxes(axs1[0][3])
    
    plt.savefig(f"plots/Plot1{label}.{format_}", format=format_, dpi=dpi)
    plt.show()
    
    #2D Hists
    fig3, axs3 = plt.subplots(3,3, figsize=(18,12), sharex="all", sharey="all", layout="compressed")

    lim_max = 10**-6
    vmins = []
    vmaxs = []
    for i, (ax, true, name) in enumerate(zip(axs3[0], data_true[7:], names[-3:])):
        name = f"<{name}>"
        bins = "log" if i == 2 else None
        im2 = ax.hexbin(data_true[0], data_true[1], C=true, gridsize=150, cmap="coolwarm", bins=bins)
        vmins.append(im2.get_array().data.min())
        vmaxs.append(im2.get_array().data.max())

        ticks_cb = np.array([1, 2, 4, 6, 8, 10, 12]) if i == 2 else None
        cbar3 = fig3.colorbar(im2, ax=axs3[:2,i], pad=0.03, aspect=33, location="bottom", shrink=0.95, ticks=ticks_cb)
        if i == 2:
            cbar3.ax.set_xticklabels(ticks_cb)
            cbar3.ax.minorticks_off()
        ax.set_title(f"Data {name}")
        lim_max = np.maximum(np.max(np.abs(ax.get_xlim()+ax.get_ylim())), lim_max)

    lim_max *= standard_zoomout

    for i, (ax, flow, name, vmin, vmax) in enumerate(zip(axs3[1], data_flow[7:], names[-3:], vmins, vmaxs)):
        name = f"<{name}>"
        include = (data_flow[0]<=lim_max)&(data_flow[1]<=lim_max)
        bins = "log" if i == 2 else None
        ax.hexbin(data_flow[0,include], data_flow[1,include], C=flow[include], gridsize=int(150*standard_zoomout), cmap="coolwarm", vmin=vmin, vmax=vmax, bins=bins)
        ax.set_title(f"Flow sample {name}")

    for i, (ax, true, flow, name) in enumerate(zip(axs3[2], data_true[7:], data_flow[7:], names[-3:])):
        name = f"<{name}>"
        include = (data_flow[0]<=lim_max)&(data_flow[1]<=lim_max)
        C_combined = np.hstack((true, flow[include]+1j))
        x_combined = np.hstack((data_true[0], data_flow[0,include]))
        y_combined = np.hstack((data_true[1], data_flow[1,include]))
        im2b = ax.hexbin(x_combined, y_combined, C=C_combined, gridsize=int(150*standard_zoomout), cmap="bwr", vmin = -2, vmax = 2, reduce_C_function=residual_combined)

        fig3.colorbar(im2b, ax=ax, pad=0.03, aspect=33, location="bottom", shrink=0.95)
        ax.set_title(f"Residual plot of {name}")

    for ax in axs3.ravel():
        ax.set_box_aspect(1)
        ax.set_xlim(-lim_max,lim_max)
        ax.set_ylim(-lim_max,lim_max)

    
    plt.savefig(f"plots/Plot2{label}.{format_}", format=format_, dpi=dpi)
    plt.show()
    
    #Histograms
    
    fig2, axs2 = plt.subplots(4,3, figsize=(18,24))
    axs2 = axs2.ravel()
    
    axs2_crop = axs2[10*[True]+2*[False]]
    for i, (ax, true, flow, name) in enumerate(zip(axs2_crop,data_true,data_flow,names)):
        ax.hist(true, bins=300, histtype="step", density=True)
        ax.hist(flow, bins=300, histtype="step", density=True)
        ax.set_xlabel(name)
        ax.set_ylabel("count")
    
    plt.delaxes(axs2[11])
    plt.delaxes(axs2[10])
    plt.savefig(f"plots/Plot3{label}.{format_}", format=format_, dpi=dpi)
    plt.show()
    
    #Cornerplot
    if True:
        every = data_flow.shape[1]//1000
        data_corner = np.hstack((data_true[:,::every], data_flow[:,::every]))
        data_dict = dict(zip(names, data_corner))
        data_dict["select"] = np.append(np.full(data_true[:,::every].shape[1],"data"),np.full(data_flow[:,::every].shape[1],"model"))
        
        sns.pairplot(pd.DataFrame(data_dict), corner=True, aspect=1, hue="select", diag_kind="kde", diag_kws ={"common_norm":False})
        
        plt.savefig(f"plots/Plot4{label}.{format_}", format=format_, dpi=dpi)
        plt.show()


def loss_plot(losses, tot_time=None, savefig=None, format="png"):
    """
    Plot the loss curve, of the training.

    Parameters
    ----------

    losses : list
        List of losses, to be plotted.
    tot_time : float, optional, default: None
        Total time of the training in minutes. If None, the x-axis will be in steps, otherwise in minutes.
    savefig : str, optional, default: None
        If not None, the plot will be saved with this in the name.
    format : str, optional, default: "png"
        Format of the saved plot.
    """
    y_axis = np.array(losses)
    x_axis = np.arange(y_axis.shape[0])/1
    if tot_time:
        x_axis *= tot_time/(y_axis.shape[0]-1)
    else:
        x_axis *= 100
    
    plt.plot(x_axis, y_axis)
    plt.xlabel("time/min" if tot_time else "step")
    plt.ylabel("<loss>$_{50}$")
    plt.title("Loss curve")
    if savefig:
        plt.savefig(f"plots/loss_{savefig}.{format}", dpi=300, format=format)
    
    plt.show()


def sortgalaxies(Galaxies, Masses):
    """
    Sort galaxies by their mass.

    Parameters
    ----------

    Galaxies : list of numpy arrays
        List of galaxies to be sorted.
    Masses : np.ndarray
        Array of masses of the galaxies. Must be the same length as Galaxies.
        Note that technically Masses could be any other quantity to sort by.

    Returns
    -------

    Galaxies : list of numpy arrays
        List of galaxies sorted by their mass.
    Masses : np.ndarray
        Array of masses of the galaxies, sorted.

    """
    order = np.argsort(Masses)
    return [Galaxies[i] for i in order], Masses[order]


def xylim(Galaxies, xylim_array, comps=(0,1)):
    Galaxies_out = []
    for galaxy, lim in zip(Galaxies,xylim_array):
        include = (lim[0] <= galaxy[:,comps[0]] <= lim[1])&(lim[2] <= galaxy[:,comps[1]] <= lim[3])
        galaxy = galaxy[include]
        Galaxies_out.append(galaxy)
    return Galaxies_out

#How many galaxies to plot per page in the plot_conditional function, if a page is plotted. ormat: (n_rows, n_columns)
page_plot_layout = (10, 7) #(8, 6)
#How many galaxyies to plot per row in the plot_conditional function, if all galaxies are plotted.
n_row_all = 4

def plot_conditional(Galaxies, Masses, type, label, show="page", scale=None, gridsize=100, cmap=None, comps=(0,1), color="global", v_pre=None, lim_pre=None):
    """
    Plot galaxies from given data sorted by mass.

    Parameters
    ----------

    Galaxies: list or iterable of array
        This contains the data of the galaxies.
    Masses: array_like
        Masses of the galaxies.
    type: {"feh", "ofe", "N"}
        Determines the type of plot. "N" means a histogram of stars. "feh" or "ofe" means average [Fe/H] or [O/Fe] map, respectivley.
    label: str
        Identifing label for naming the saved file.
    show: {"page", "all"}, default: "page"
        Weather to plot a preset number of galaxies specified by page_plot_layout (default 70) to be fitting on an page and saved as pdf.
        Or to plot all galaxies in larger scale and save as png. In the former case galaxies of median masses get taken out while highest and
        lowest masses are alway plotted.
    scale: None or {"lin", "log"}, default: None
        Color scaling of the plots. ``None`` will result in linear scaling for [O/Fe] and [Fe/H] and log scaling for N.
    gridsize: int or (int, int), default: 100
        Gridsize for binned plot.
    cmap: None or str, default: None
        Colormap to be used in the plots. ``None`` means "magma" for N, and "coolwarm" otherwise.
    comps: (0 <= int < 3, 0 <= int < 3), default: (0, 1)
        Components to plot on the x and y axis, respectivley.
    color: {"global", "individual"}, default: "global"
        Weather all subplots share the same color scale, or all subplots scale on their individual maxima and minima. The former returns
        the used scale for reuse as pre-set values in another plot for better comparison.
    v_pre: None or (float, float), default: None
        Preset minimum and maximum value for scaling color. Applies to all plots. `None` means to calculate the scaling values from the data. See ``color``.
    lim_pre: None or np.ndarray, default: None
        Gives the limits for the x and y axis for each galaxy. Shape: (len(Galaxies), 4), where the 4 values for each galaxy are x_min, x_max, y_min, y_max.
        If `None`, the limits are calculated from the data.
        Presets not only the x, y limits of the plot, but also the limits of the data used for the plot. This is to avoid large differences in the gridsize used for the plot.

    Returns
    -------

    v_pre: None or (float, float)
        If ``color`` is "global" and ``v_pre`` is `None`, the used scaling values are returned for reuse as pre-set values in another plot for better comparison.
        For input as ``v_pre``.
    lim_pre_new: None or np.ndarray
        If ``lim_pre`` is `None`, the used limits are returned for reuse as pre-set values in another plot for better comparison.
        For input as ``lim_pre``.
    """

    #Standard colormap
    if cmap == None:
        cmap = "magma" if type == "N" else "coolwarm"

    #Initialize
    #Data
    Galaxies_sorted, Masses_sorted = sortgalaxies(Galaxies, Masses)
    if show == "page":
        N_plot = np.prod(page_plot_layout)
        N_galaxy = len(Galaxies)
        N_leavout = N_galaxy-N_plot
        N_remain = N_galaxy-N_leavout*2
        plot_galaxy = [True]*(N_remain//2) + [True, False]*(N_leavout) + [True]*(N_remain-N_remain//2)
        Galaxies_sorted = [Galaxies_sorted[i] for i in np.arange(N_galaxy)[plot_galaxy]]
        Masses_sorted = Masses_sorted[plot_galaxy]
        plot_layout = page_plot_layout
        figsize = (8.27, 11.69)
    else:
        plot_layout = (-(len(Galaxies)//-n_row_all), n_row_all)
        figsize = (16, 4*plot_layout[0])
    
    #Sclaing log/lin
    if scale is None:
        scale = "log" if type == "N" else "lin"
    bins = "log" if scale == "log" else None

    #Get global scaling values
    if color == "global" and v_pre == None:
        vmax = -100
        vmin = 100
        for galaxy in Galaxies_sorted:
            if type == "ofe":
                statistic = galaxy[:,7]
            elif type == "feh":
                statistic = galaxy[:,8]
            else:
                statistic = None
        
            res = plt.hexbin(galaxy[:,comps[0]], galaxy[:,comps[1]], C=statistic, bins=bins, gridsize=gridsize, cmap=cmap, rasterized=True)
            plt.close()
            bin_results = res.get_array().data
            vmax = np.maximum(bin_results.max(), vmax)
            vmin = np.minimum(bin_results.min(), vmin)
        vmin = 1 if type=="N" else vmin
    elif color == "global":
        vmin, vmax = v_pre
    else:
        vmin, vmax = None, None

    #Plot layout
    fig, axs = plt.subplots(*plot_layout, figsize=figsize, layout="constrained")
    axs = axs.ravel()

    #Initialize saving new limits
    if lim_pre is None:
        lim_pre_new = np.zeros((len(Galaxies_sorted),4))

    #Make plots
    for i, (ax, galaxy, mass) in enumerate(zip(axs, Galaxies_sorted, Masses_sorted)):

        #Apply preset x/y limits on data
        if lim_pre is not None:
            include = (lim_pre[i][0]<=galaxy[:,comps[0]])&(galaxy[:,comps[0]]<=lim_pre[i][1])&(lim_pre[i][2]<=galaxy[:,comps[1]])&(galaxy[:,comps[1]]<=lim_pre[i][3])
            gridsize = int(gridsize*standard_zoomout/standard_zoomout)
        else:
            include = np.full(galaxy.shape[0], True)
        ic = include

        #Prepare right type to be plotted
        if type == "ofe":
            statistic = galaxy[ic,7]
        elif type == "feh":
            statistic = galaxy[ic,8]
        else:
            statistic = None

        im = ax.hexbin(galaxy[ic,comps[0]], galaxy[ic,comps[1]], C=statistic, bins=bins, gridsize=gridsize, cmap=cmap, rasterized=True, vmin=vmin, vmax=vmax)
        
        ax.set_title(f"M = {mass:.2e}M$_\odot$", fontsize=5, pad=1)
        ax.set_xlabel(comp_names[comps[0]], fontsize=5, labelpad=0.2)
        ax.set_ylabel(comp_names[comps[1]], fontsize=5, labelpad=0.1)

        #Facecolor
        if type=="N":
            ax.set_facecolor(matplotlib.colormaps[cmap](0))

        #No ticks
        #ax.set_xticks([])
        #ax.set_yticks([])

        #Small, fitting ticks
        ax.tick_params(axis="both", labelsize=5, length=2.5, pad=0.5)

        ax.set_aspect("equal")
        ax.set_box_aspect(1)

        #Get x/y lims to be set and potentially saved
        if lim_pre is None:
            s_z = standard_zoomout
            x_lim_new = (ax.get_xlim()[0]*s_z, ax.get_xlim()[1]*s_z)
            y_lim_new = (ax.get_ylim()[0]*s_z, ax.get_ylim()[1]*s_z)
            lim_pre_new[i] = np.array([*x_lim_new,*y_lim_new])
        else:
            x_lim_new = tuple(lim_pre[i][:2])
            y_lim_new = tuple(lim_pre[i][-2:])
        
        ax.set_xlim(x_lim_new)
        ax.set_ylim(y_lim_new)

    #Whole figure title + colorbar
    fig.suptitle(f'{"<[O/Fe]>" if type == "ofe" else ("<[Fe/H]>" if type == "feh" else "<N>")} in dependency of total mass M')
    if color == "global":
        plt.colorbar(im, ax=axs, shrink = 0.95, location="bottom", aspect=50, pad=0.02)

    #Delete axis left over
    n_not_used = len(Galaxies_sorted)-plot_layout[0]*plot_layout[1]
    if n_not_used<0:
        for not_used in axs[n_not_used:]:
            plt.delaxes(not_used)

    #Save and show
    format = "pdf" if show == "page" else "png"
    plt.savefig(f"plots/Plot_conditional_{label}.{format}", dpi=300, format=format)
    plt.show()

    #Return scale values
    if color == "global" and v_pre == None and lim_pre is None:
        return vmin, vmax, lim_pre_new
    elif color == "global" and v_pre == None:
        return vmin, vmax
    


#Plot histograms for conditional data
#Histogram for each property r,z,abs(v),Z,FeH,OFe, age. The galaxies are color coded by their mass M, and in the same histogram.
#Coloring is done by sampling from a colormap in log, and the colorbar is placed at the bottom of the figure.

def plot_conditional_histograms(Galaxies, Massses, label, bins=100, cmap="magma", log=False):
    colormap = matplotlib.colormaps[cmap]
    c_norm = matplotlib.colors.LogNorm(vmin=Massses.min(), vmax=Massses.max())
    scalar_map = matplotlib.cm.ScalarMappable(norm=c_norm, cmap=colormap)

    plottables = ["r/kpc", "z/kpc", "|v|/km/s", "Z", "[Fe/H]", "[O/Fe]", "age/Gyr"]

    fig, axs = plt.subplots(3,3, figsize=(9,21), layout="constrained")
    axs = axs.ravel()

    for galaxy, mass in zip(Galaxies, Massses):
        for i, (ax, name) in enumerate(axs, plottables):
            if i==0:
                #Get cylindrical radius
                plot = np.sqrt(np.sum(galaxy[:,:2]**2, axis=1))
            elif i==1:
                #Get z
                plot = galaxy[:,2]
            elif i==2:
                #Get absolute velocity
                plot = np.sqrt(np.sum(galaxy[:,3:6]**2, axis=1))
            elif i>=3:
                ind_plot = i+3
                plot = galaxy[:,ind_plot]
            
            ax.hist(plot, bins=bins, color=scalar_map.to_rgba(mass), density=True, histtype="step", log=log)
            ax.set_xlabel(name)
            ax.set_ylabel("Density")
            
    fig.suptitle("Histograms of the properties of the galaxies, colored by mass")
    plt.colorbar(scalar_map, ax=axs, shrink = 0.95, location="bottom", aspect=50, pad=0.02)
    plt.savefig(f"plots/Cond_histograms_{label}.pdf", dpi=300, format="pdf")
    plt.show()


#Rewrite plot_conditional to use scipy binned_statistic_2d and plt.imshow

def plot_conditional_2(*Data_colection ,type="N", label="", show="page", scale=None, gridsize=100, cmap=None, comps=(0,1), color="global", color_pass="local", grid_pass=1, N_density=True):
    #Standard colormap
    if cmap == None:
        cmap = "magma" if type == "N" else "coolwarm"

    #Sclaing log/lin
    if scale is None:
        scale = "log" if type == "N" else "lin"

    #Initialize
    #Data
    Galaxies_col_sorted = []
    Masses_col_sorted = []
    figsizes = []
    for Galaxies, Masses in zip(Data_colection[::2], Data_colection[1::2]):
        Galaxies_sorted, Masses_sorted = sortgalaxies(Galaxies, Masses)
        if show == "page":
            N_plot = np.prod(page_plot_layout)
            N_galaxy = len(Galaxies)
            N_leavout = N_galaxy-N_plot
            N_remain = N_galaxy-N_leavout*2
            plot_galaxy = [True]*(N_remain//2) + [True, False]*(N_leavout) + [True]*(N_remain-N_remain//2)
            Galaxies_sorted = [Galaxies_sorted[i] for i in np.arange(N_galaxy)[plot_galaxy]]
            Masses_sorted = Masses_sorted[plot_galaxy]
            plot_layout = page_plot_layout
            figsizes.append((8.27, 11.69))
        else:
            plot_layout = (-(len(Galaxies)//-n_row_all), n_row_all)
            figsizes.append((16, 4*plot_layout[0]))

        Galaxies_col_sorted.append(Galaxies_sorted)
        Masses_col_sorted.append(Masses_sorted)

    #Statistics
    if type == "N":
        statistic = "count"
    elif type == "ofe":
        statistic = "mean"
    elif type == "feh":
        statistic = "mean"

    #Iterate over all datasets of galaxies and calculate all statistics
    #Then use the results to set scalings (color, grid etc.)
    #Either use the same scalings for all datasets, or use different scalings for each dataset
    #Behaviour following the color_pass and grid_pass arguments
    #color_pass = "local" -> use local color scalings for each dataset
    #color_pass = "global" -> use global color scalings for all datasets
    #grind_pass determines how many times the grid is passed over from the first dataset

    Result_col = []
    vmin_s = []
    vmax_s = []
    for i, (Galaxies_sorted, Masses_sorted) in enumerate(zip(Galaxies_col_sorted, Masses_col_sorted)):
        #Initialize
        Result = []
        vmin = np.inf
        vmax = -np.inf
        #Iterate over all galaxies
        for j, (galaxy, mass) in enumerate(zip(Galaxies_sorted, Masses_sorted)):
            #Get right grid
            if i==0 or i>grid_pass:
                #Do individual grid, with quadratic bins
                sz = standard_zoomout
                x_bins = np.linspace(galaxy[:,comps[0]].min()*sz, galaxy[:,comps[0]].max()*sz, gridsize)
                y_bins = np.linspace(galaxy[:,comps[1]].min()*sz, galaxy[:,comps[1]].max()*sz, gridsize)
            else:
                #Use grid from first dataset from respective galaxy
                x_bins = Result_col[0][j][1]
                y_bins = Result_col[0][j][2]


            #Calculate statistics
            if type == "N":
                result = binned_statistic_2d(galaxy[:,comps[0]], galaxy[:,comps[1]], galaxy[:,0], statistic="count", bins=(x_bins, y_bins))
            elif type == "ofe":
                result = binned_statistic_2d(galaxy[:,comps[0]], galaxy[:,comps[1]], galaxy[:,8], statistic="mean", bins=(x_bins, y_bins))
            elif type == "feh":
                result = binned_statistic_2d(galaxy[:,comps[0]], galaxy[:,comps[1]], galaxy[:,7], statistic="mean", bins=(x_bins, y_bins))

            vmin = min(vmin, result[0].min())
            vmax = max(vmax, result[0].max())

            Result.append(result)
        Result_col.append(Result)

        #Set scalings for color
        if color_pass == "local":
            #Use individual scaling for each dataset
            vmin_s.append(vmin)
            vmax_s.append(vmax)
        elif color_pass == "global":
            #Use global scaling for all datasets
            vmin_s.append(min(min(vmin_s), vmin))
            vmax_s.append(max(max(vmax_s), vmax))

    #Plot
    for i, (Galaxies_sorted, Masses_sorted, Result, figsize, vmin, vmax) in enumerate(zip(Galaxies_col_sorted, Masses_col_sorted, Result_col, figsizes, vmin_s, vmax_s)):
        fig, axs = plt.subplots(*plot_layout, figsize=figsize, layout="constrained")
        axs = axs.ravel()
        for ax, galaxy, mass, result in zip(axs, Galaxies_sorted, Masses_sorted, Result):

            #Respect scaling log/lin
            statistic = result[0]
            if type == "N" and N_density:
                #Get area of bins in units pc^2
                area = (result[1][1]-result[1][0])*(result[2][1]-result[2][0])*1e6
                #Scale statistic to be per pc^2
                statistic = statistic/area

            if color == "individual":
                #Use individual color scaling for each galaxy
                vmin = statistic.min()
                vmax = statistic.max()

            vmin = 1 if vmin == 0 else vmin

            if scale == "log":
                #Logarithmic scaling
                norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
            elif scale == "lin":
                #Linear scaling
                norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

            #Plot
            im = ax.imshow(statistic.T, origin="lower", extent=[result[1].min(), result[1].max(), result[2].min(), result[2].max()], cmap=cmap, norm=norm)

            #Set labels
            ax.set_title(f"M = {mass:.2e}M$_\odot$", fontsize=5, pad=1)
            ax.set_xlabel(comp_names[comps[0]], fontsize=5, labelpad=0.2)
            ax.set_ylabel(comp_names[comps[1]], fontsize=5, labelpad=0.1)

            #Facecolor
            if type=="N":
                ax.set_facecolor(matplotlib.colormaps[cmap](0))

            #No ticks
            #ax.set_xticks([])
            #ax.set_yticks([])

            #Small, fitting ticks
            ax.tick_params(axis="both", labelsize=5, length=2.5, pad=0.5)

            ax.set_aspect("equal")
            ax.set_box_aspect(1)

        #Colorbar, suptitle
        #Whole figure title + colorbar
        fig.suptitle(f'{"<[O/Fe]>" if type == "ofe" else ("<[Fe/H]>" if type == "feh" else "N")} in dependency of total mass M')
        if color == "global":
            fig.colorbar(im, ax=axs, shrink = 0.95, location="bottom", aspect=50, pad=0.02)

        #Delete axis left over
        n_not_used = len(Galaxies_sorted)-plot_layout[0]*plot_layout[1]
        if n_not_used<0:
            for not_used in axs[n_not_used:]:
                fig.delaxes(not_used)

        #Save and show
        format = "pdf" if show == "page" else "png"
        fig.savefig(f"plots/Plot_conditional_{label}_{i}.{format}", dpi=300, format=format)
        fig.show()

            
            


    
    


