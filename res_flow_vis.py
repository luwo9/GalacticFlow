import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from scipy.stats import binned_statistic_2d as bs2d



def get_result_plots(data_true_, data_flow_, label="", format_="png", dpi=300):
    
    data_true = data_true_.T
    data_flow = data_flow_.T
    names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'metals', '[Fe/H]', '[O/Fe]', 'age/Gyr']
    
    
    #2D Plots xyz corner
    fig1, axs1 = plt.subplots(2,2, sharex = "col", sharey = False, figsize=(12,12))
    axs1[0][0].scatter(data_true[0], data_true[1], label = "Data", s=0.2)
    axs1[0][0].scatter(data_flow[0], data_flow[1], label = "Flow", s=0.2)
    axs1[0][0].set_xlabel("x")
    axs1[0][0].set_ylabel("y")
    axs1[0][0].set_title("Scatter plot in xy-plane")

    axs1[1][0].scatter(data_true[0], data_true[2], label = "Data", s=0.2)
    axs1[1][0].scatter(data_flow[0], data_flow[2], label = "Flow", s=0.2)
    axs1[1][0].set_xlabel("x")
    axs1[1][0].set_ylabel("z")
    axs1[1][0].set_title("Scatter plot in xz-plane")

    axs1[1][1].scatter(data_true[1], data_true[2], label = "Data", s=0.2)
    axs1[1][1].scatter(data_flow[1], data_flow[2], label = "Flow", s=0.2)
    axs1[1][1].set_xlabel("y")
    axs1[1][1].set_ylabel("z")
    axs1[1][1].set_title("Scatter plot in yz-plane")
    
    for ax in axs1.ravel():
        ax.set_aspect("equal", adjustable="datalim")
    plt.delaxes(axs1[0][1])
    
    plt.savefig(f"plots/Plot1{label}.{format_}", format=format_, dpi=dpi)
    plt.show()
    
    #2D Hists
    fig3, axs3 = plt.subplots(2,4, figsize=(18,9))
    maxr = np.abs(data_true[:2]).max()*1.5
    is_inside_maxr = (data_flow[0]<=maxr)&(data_flow[1]<=maxr)
    for i, (ax, true, flow, name) in enumerate(zip(axs3.T, data_true[-4:], data_flow[-4:], names[-4:])):

        name = f"<{name}>"
        #Exlude non clear cuts for similar color and values to far out for similar resolution
        if i==0:
            name = "log<N>"
            true = None
            flow1 = None
            cond_corr = is_inside_maxr
            bins = "log"
            ax[0].set_facecolor((0.2298057, 0.298717966, 0.753683153, 1.0))
            ax[1].set_facecolor((0.2298057, 0.298717966, 0.753683153, 1.0))
        else:
            cond_corr = (flow<=true.max())&(flow>=true.min())&(is_inside_maxr)
            flow1 = flow[cond_corr]
            bins = None

        '''binned = bs2d(data_true[0], data_true[1], values=true, bins=100)
        ax[0].pcolormesh(binned.x_edge, binned.y_edge, binned.statistic, cmap="coolwarm")'''
        ax[0].hexbin(data_true[0], data_true[1], C=true, gridsize=150, cmap="coolwarm", bins=bins)
        ax[0].set_title(f"Data {name}")


        '''binned = bs2d(data_flow[0,cond_corr], data_flow[1,cond_corr], values=flow1, bins=100)
        ax[1].pcolormesh(binned.x_edge, binned.y_edge, binned.statistic, cmap="coolwarm")'''
        ax[1].hexbin(data_flow[0,cond_corr], data_flow[1,cond_corr], C=flow1, gridsize=int(150*1.5), cmap="coolwarm", bins=bins)
        ax[1].set_title(f"Flow sample {name}")
        for axi in ax:
            axi.set_xlabel("x")
            axi.set_ylabel("y")
            axi.set_xlim(-maxr,maxr)
            axi.set_ylim(-maxr,maxr)


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
        data_corner = np.hstack((data_true[:,::700], data_flow[:,::700]))
        data_dict = dict(zip(names, data_corner))
        data_dict["select"] = np.append(np.full(data_true[:,::700].shape[1],"data"),np.full(data_flow[:,::700].shape[1],"model"))
        
        sns.pairplot(pd.DataFrame(data_dict), corner=True, aspect=1, hue="select", diag_kind="kde")
        
        plt.savefig(f"plots/Plot4{label}.{format_}", format=format_, dpi=dpi)
        plt.show()


def loss_plot(losses, tot_time=None, savefig=None, format="png"):
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
        plt.savefig(f"plots/{savefig}.{format}", dpi=300, format=format)
    
    plt.show()


def sortgalaxies(Galaxies, Masses):
    order = np.argsort(Masses)
    return [Galaxies[i] for i in order], Masses[order]


page_plot_layout = (10, 7) #(8, 6)
n_row_all = 4
comp_names = "xyz"

def plot_conditional(Galaxies, Masses, type, label, show="page", scale=None, gridsize=100, cmap=None, comps=(0,1), color="global", v_pre=None):
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

    #Make plots
    for ax, galaxy, mass in zip(axs, Galaxies_sorted, Masses_sorted):
        if type == "ofe":
            statistic = galaxy[:,7]
        elif type == "feh":
            statistic = galaxy[:,8]
        else:
            statistic = None

        im = ax.hexbin(galaxy[:,comps[0]], galaxy[:,comps[1]], C=statistic, bins=bins, gridsize=gridsize, cmap=cmap, rasterized=True, vmin=vmin, vmax=vmax)
        
        ax.set_title(f"M = {mass:.2e}", fontsize=5, pad=1)
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
        ax.set_adjustable("datalim")
        ax.set_aspect("equal")

    #Whole figure title + colorbar
    fig.suptitle(f'{"<[O/Fe]>" if type == "ofe" else ("<[Fe/H]>" if type == "feh" else "<N>")} in dependency of M')
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
    if color == "global" and v_pre == None:
        return vmin, vmax