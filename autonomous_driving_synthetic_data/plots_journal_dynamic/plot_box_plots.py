import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections
import seaborn as sns
from pylab import setp
import argparse

sns.set_theme(style = "whitegrid", palette = 'tab10')
matplotlib.rc('xtick', labelsize=30)
matplotlib.rc('ytick', labelsize=30)
matplotlib.rc('font', weight='bold')
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['savefig.format'] = "pdf"

# function for setting the colors of the box plots pairs
def setBoxColors(bp):
    lw=5
    setp(bp['boxes'][0], color='red',linewidth=lw)
    setp(bp['medians'][0], color='orange',linewidth=lw)

    setp(bp['boxes'][1], color='cyan',linewidth=lw)
    setp(bp['medians'][1], color='orange',linewidth=lw)

    setp(bp['boxes'][2], color='blue',linewidth=lw)
    setp(bp['medians'][2], color='orange',linewidth=lw)

parser = argparse.ArgumentParser()
parser.add_argument("--num_obs",  type=int, required=True)
parser.add_argument('--setting',type=str, nargs='+', required=True)
parser.add_argument('--num_reduced_set',type=int, nargs='+', required=True)

args = parser.parse_args()

showfliers= False
num_reduced_list = args.num_reduced_set
num_obs = args.num_obs

list_setting = args.setting

for sc in list_setting:
    num = len(num_reduced_list) 

    fig, axs = plt.subplots(num,1, figsize=(12, 6))

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)

    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    plt.ylabel('% Collisions', fontweight = "bold", fontsize = 30)

    for i,num_reduced in enumerate(num_reduced_list):
        filename = "../stats/dynamic/{}/stats_{}_samples_{}_obs.npz".format(sc,num_reduced,num_obs)

        temp_file = np.load(filename)

        coll_mmd = 100*(temp_file["coll_mmd"])/1000
        coll_saa = 100*(temp_file["coll_saa"])/1000
        coll_cvar = 100*(temp_file["coll_cvar"])/1000
        
        if len(coll_cvar)==0 or len(coll_mmd)==0 or len(coll_saa)==0:
            continue

        data = [coll_mmd,coll_saa,coll_cvar]

        x_synthetic = np.array([0.5,1.5,2.5]) # the label locations
        widths = 0.8
        width= 1.0
        whiskerprops = dict(linestyle='-',linewidth=5.0, color='black')

        bp = axs[i].boxplot(data,showfliers=showfliers,widths=widths,whiskerprops=whiskerprops)
        setBoxColors(bp)

        axs[i].set_xticklabels([])

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        textstr = "{} samples".format(num_reduced)
        axs[i].text(0.05, 0.95, textstr, transform=axs[i].transAxes, fontsize=30,
        verticalalignment='top', bbox=props)

    labels_synthetic = ["$r_{MMD}^{emp}$","$r_{SAA}$","$r_{CVaR}^{emp}$"]
    axs[i].set_xticks(x_synthetic + width / 2, labels_synthetic)

    fig.tight_layout(pad=0.5)

    plt.show()
