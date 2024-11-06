import sys
sys.path.insert(1, '/home/ims-robotics/MMD-OPT/autonomous_driving_synthetic_data')
sys.path.insert(1, '/home/ims-robotics/MMD-OPT/autonomous_driving_synthetic_data/optimizer')
from optimizer import cem
import numpy as np
import matplotlib.pyplot as plt
import jax
import scipy
import matplotlib.patches as pt
import matplotlib
import seaborn as sns
import argparse
from obs_data_generate_static import compute_obs_trajectories

fs = 13
sns.set_theme(style = "whitegrid", palette = 'tab10')
matplotlib.rc('xtick', labelsize=fs)
matplotlib.rc('ytick', labelsize=fs)
matplotlib.rc('font', weight='bold')
matplotlib.rcParams['savefig.format'] = "pdf"

a_min,a_max,steer_min,steer_max = -18.,18.,-0.6,0.6

def normal_vectors(x, y, scalar):
    tck = scipy.interpolate.splrep(x, y)
    y_deriv = scipy.interpolate.splev(x, tck, der=1)
    normals_rad = np.arctan(y_deriv)+np.pi/2.
    return np.cos(normals_rad)*scalar, np.sin(normals_rad)*scalar

def compute_f_bar_temp(x_obs,y_obs,x,y): 
    
    wc_alpha = (x-x_obs[:,0:prob0.num_prime])
    ws_alpha = (y-y_obs[:,0:prob0.num_prime])

    cost = -(wc_alpha**2)/(prob0.a_obs**2) - (ws_alpha**2)/(prob0.b_obs**2) + np.ones((prob0.num_circles*_num_batch,prob0.num_prime))

    costbar = np.maximum(np.zeros((prob0.num_circles*_num_batch,prob0.num_prime)),cost)
        
    return costbar
 
def compute_stats(seed,cx,cy,x_obs,y_obs):

    x_obs = x_obs.reshape(-1)
    y_obs = y_obs.reshape(-1)

    # print(x_obs,y_obs)
    psi_obs = np.zeros_like(x_obs)
    
    X_obs1,Y_obs1,Psi_obs1 =  np.zeros((prob0.num_obs,_num_batch*prob0.num_circles,prob0.num)),\
np.zeros((prob0.num_obs,_num_batch*prob0.num_circles,prob0.num)),np.zeros((prob0.num_obs,_num_batch*prob0.num_circles,prob0.num))
    
    x = np.dot(prob0.P_jax,cx.reshape(-1))
    y = np.dot(prob0.P_jax,cy.reshape(-1))

    count = np.zeros(prob0.num_obs)
   
    for tt in range(num_obs):
        _x_obs_init,_y_obs_init,_psi_obs_init = compute_noisy_init_state(x_obs[tt],y_obs[tt],psi_obs[tt],_num_batch,seed)
        _x_obs_traj,_y_obs_traj,_psi_obs_traj = compute_obs_trajectories(_x_obs_init,_y_obs_init,_psi_obs_init)       
        
        cost = compute_f_bar_temp(_x_obs_traj,_y_obs_traj,x,y) # _num_batch x num
        cost = cost.T # num X _num_batch
       
        intersection = np.count_nonzero(cost,axis=1) # num 
        
        count[tt]=np.max(intersection)

        X_obs1[tt], Y_obs1[tt], Psi_obs1[tt] = _x_obs_traj, _y_obs_traj, _psi_obs_traj
    
    # print("count", count)
    return np.max(count),X_obs1,Y_obs1,Psi_obs1

_num_batch = 10**3 

parser = argparse.ArgumentParser()
parser.add_argument("--num_obs",  type=int, required=True)
parser.add_argument('--num_reduced_set',type=int, required=True)
parser.add_argument('--noises',type=str, nargs='+', required=True)
parser.add_argument("--num_exps",  type=int, required=True)
parser.add_argument('--costs',type=str, nargs='+', required=True)

args = parser.parse_args()
num_reduced = args.num_reduced_set
num_mother = _num_batch
num_obs = args.num_obs # number of nearest obs to take into account
num_exps = args.num_exps
noises = args.noises

for noise in noises:
    if noise == "gaussian":
        from obs_data_generate_static import compute_noisy_init_state_gaussian as compute_noisy_init_state
    elif noise == "bimodal":
        from obs_data_generate_static import compute_noisy_init_state_bimodal as compute_noisy_init_state
    else:
        from obs_data_generate_static import compute_noisy_init_state_trimodal as compute_noisy_init_state

    root = "../data/static/{}".format(noise)

    prob0 = cem.CEM(num_reduced,num_mother,num_obs,"static")

    dt = prob0.t

    key0 = prob0.key
    num = prob0.num_prime
    v_min = prob0.v_min
    v_max = prob0.v_max

    coll_mmd = []
    coll_saa = []
    coll_cvar = []
    coll_rand = []

    mmd_set_all,mmd_rand_set_all,cvar_set_all,saa_set_all = [],[],[],[]
    
    for l in range(num_exps):
        data_saa = np.load(root + "/saa_{}_samples_{}_obs_{}.npz".format(num_reduced,num_obs,l))
        data_cvar = np.load(root + "/cvar_{}_samples_{}_obs_{}.npz".format(num_reduced,num_obs,l))
        data = np.load(root + "/mmd_opt_{}_samples_{}_obs_{}.npz".format(num_reduced,num_obs,l))
    
        cx_all = np.asarray(data["cx"])
        cy_all = np.asarray(data["cy"])
        x_obs =  np.asarray(data["x_obs"])
        y_obs =  np.asarray(data["y_obs"])
        
        cx_all_saa = np.asarray(data_saa["cx"])
        cy_all_saa = np.asarray(data_saa["cy"])
        x_obs_saa =  np.asarray(data_saa["x_obs"])
        y_obs_saa =  np.asarray(data_saa["y_obs"])

        cx_all_cvar = np.asarray(data_cvar["cx"])
        cy_all_cvar = np.asarray(data_cvar["cy"])
        x_obs_cvar =  np.asarray(data_cvar["x_obs"])
        y_obs_cvar =  np.asarray(data_cvar["y_obs"])

        mmd_matrix = np.hstack((x_obs[:,0:num_obs],y_obs[:,0:num_obs]))
        mmd_set = set([tuple(x) for x in mmd_matrix])
        mmd_set_all.append(mmd_set)

        saa_matrix = np.hstack((x_obs_saa[:,0:num_obs],y_obs_saa[:,0:num_obs]))
        saa_set = set([tuple(x) for x in saa_matrix])
        saa_set_all.append(saa_set)

        cvar_matrix = np.hstack((x_obs_cvar[:,0:num_obs],y_obs_cvar[:,0:num_obs]))
        cvar_set = set([tuple(x) for x in cvar_matrix])
        cvar_set_all.append(cvar_set)
    
    mmd_set_final,mmd_rand_set_final,saa_set_final,cvar_set_final = np.zeros((0,cvar_matrix.shape[1])),\
                                                                np.zeros((0,cvar_matrix.shape[1])),np.zeros((0,cvar_matrix.shape[1])),\
                                                                np.zeros((0,cvar_matrix.shape[1]))
    
    for x in mmd_set_all[0]:
        count  = 0
        for pp in range(1,num_exps):
            if x in mmd_set_all[pp]:
                count += 1
        if count == num_exps-1:
            x = np.array([x])
            mmd_set_final = np.vstack((mmd_set_final,x))
    
    for x in saa_set_all[0]:
        count  = 0
        for pp in range(1,num_exps):
            if x in saa_set_all[pp]:
                count += 1
        if count == num_exps-1:
            x = np.array([x])
            saa_set_final = np.vstack((saa_set_final,x))

    for x in cvar_set_all[0]:
        count  = 0
        for pp in range(1,num_exps):
            if x in cvar_set_all[pp]:
                count += 1
        if count == num_exps-1:
            x = np.array([x])
            cvar_set_final = np.vstack((cvar_set_final,x))

    temp2 = set([tuple(x) for x in saa_set_final])
    temp3 = set([tuple(x) for x in cvar_set_final])
    temp4 = set([tuple(x) for x in mmd_set_final])

    common_set = np.array([x for x in temp2 & temp3 & temp4])
    print(common_set.shape[0])

    cx_all_mmd_all,cy_all_mmd_all,init_state_all_mmd_all,\
        x_obs_mmd_all,y_obs_mmd_all,psi_obs_mmd_all\
    = np.zeros((num_exps,common_set.shape[0],prob0.nvar)),\
        np.zeros((num_exps,common_set.shape[0],prob0.nvar)),\
            np.zeros((num_exps,common_set.shape[0],6)),\
                np.zeros((num_exps,common_set.shape[0],num_obs)),\
                    np.zeros((num_exps,common_set.shape[0],num_obs)),\
                        np.zeros((num_exps,common_set.shape[0],num_obs)) # num_exps x num_obs_configs x _

    cx_all_saa_all,cy_all_saa_all,init_state_all_saa_all,x_obs_saa_all,y_obs_saa_all,psi_obs_saa_all\
    = np.zeros((num_exps,common_set.shape[0],prob0.nvar)),\
        np.zeros((num_exps,common_set.shape[0],prob0.nvar)),\
            np.zeros((num_exps,common_set.shape[0],6)),\
                np.zeros((num_exps,common_set.shape[0],num_obs)),\
                    np.zeros((num_exps,common_set.shape[0],num_obs)),\
                        np.zeros((num_exps,common_set.shape[0],num_obs)) # num_exps x num_obs_configs x _

    cx_all_cvar_all,cy_all_cvar_all,init_state_all_cvar_all,x_obs_cvar_all,y_obs_cvar_all,psi_obs_cvar_all\
    = np.zeros((num_exps,common_set.shape[0],prob0.nvar)),\
        np.zeros((num_exps,common_set.shape[0],prob0.nvar)),\
            np.zeros((num_exps,common_set.shape[0],6)),\
                np.zeros((num_exps,common_set.shape[0],num_obs)),\
                    np.zeros((num_exps,common_set.shape[0],num_obs)),\
                        np.zeros((num_exps,common_set.shape[0],num_obs)) # num_exps x num_obs_configs x _

    for l in range(num_exps):
        data_saa = np.load(root + "/saa_{}_samples_{}_obs_{}.npz".format(num_reduced,num_obs,l))
        data_cvar = np.load(root + "/cvar_{}_samples_{}_obs_{}.npz".format(num_reduced,num_obs,l))
        data = np.load(root + "/mmd_opt_{}_samples_{}_obs_{}.npz".format(num_reduced,num_obs,l))
    
        cx_all = np.asarray(data["cx"])
        cy_all = np.asarray(data["cy"])
        x_obs =  np.asarray(data["x_obs"])
        y_obs =  np.asarray(data["y_obs"])

        cx_all_saa = np.asarray(data_saa["cx"])
        cy_all_saa = np.asarray(data_saa["cy"])
        x_obs_saa =  np.asarray(data_saa["x_obs"])
        y_obs_saa =  np.asarray(data_saa["y_obs"])

        cx_all_cvar = np.asarray(data_cvar["cx"])
        cy_all_cvar = np.asarray(data_cvar["cy"])
        x_obs_cvar =  np.asarray(data_cvar["x_obs"])
        y_obs_cvar =  np.asarray(data_cvar["y_obs"])
        
        mmd_matrix = np.hstack((x_obs[:,0:num_obs],
                        y_obs[:,0:num_obs]))
        
        saa_matrix = np.hstack((x_obs_saa[:,0:num_obs],y_obs_saa[:,0:num_obs]))
        cvar_matrix = np.hstack((x_obs_cvar[:,0:num_obs],y_obs_cvar[:,0:num_obs]))
    
        for k in range(common_set.shape[0]):

            idx_mmd = np.where(np.all(common_set[k]==mmd_matrix,axis=1))
            if len(idx_mmd[0]) > 1 :
                idx_mmd = idx_mmd[0][0]

            cx_all_mmd_all[l][k] = cx_all[idx_mmd] 
            cy_all_mmd_all[l][k] = cy_all[idx_mmd] 
            x_obs_mmd_all[l][k] = x_obs[idx_mmd] 
            y_obs_mmd_all[l][k] = y_obs[idx_mmd] 
            
            idx_saa = np.where(np.all(common_set[k]== saa_matrix,axis=1))
            if len(idx_saa[0]) > 1 :
                idx_saa = idx_saa[0][0]

            cx_all_saa_all[l][k] = cx_all_saa[idx_saa] 
            cy_all_saa_all[l][k] = cy_all_saa[idx_saa] 
            x_obs_saa_all[l][k] = x_obs_saa[idx_saa] 
            y_obs_saa_all[l][k] = y_obs_saa[idx_saa] 
            
            idx_cvar = np.where(np.all(common_set[k]==cvar_matrix,axis=1))
            if len(idx_cvar[0]) > 1 :
                idx_cvar = idx_cvar[0][0]

            cx_all_cvar_all[l][k] = cx_all_cvar[idx_cvar] 
            cy_all_cvar_all[l][k] = cy_all_cvar[idx_cvar] 
            x_obs_cvar_all[l][k] = x_obs_cvar[idx_cvar] 
            y_obs_cvar_all[l][k] = y_obs_cvar[idx_cvar] 
            
    num_p = 25000
    x_path = np.linspace(0,1000,num_p)
    y_path = np.zeros(num_p)

    x_path_normal_lb,y_path_normal_lb = normal_vectors(x_path,y_path,-3.5)
    x_path_lb = x_path + x_path_normal_lb
    y_path_lb = y_path + y_path_normal_lb

    x_path_normal_ub,y_path_normal_ub = normal_vectors(x_path,y_path,3.5)
    x_path_ub = x_path + x_path_normal_ub
    y_path_ub = y_path + y_path_normal_ub

    x_path_normal_d_mid,y_path_normal_d_mid = normal_vectors(x_path,y_path,0)
    x_path_d_mid = x_path + x_path_normal_d_mid
    y_path_d_mid = y_path + y_path_normal_d_mid

    len_path = 6000
    linewidth = 0.5

    for i in range(num_exps):
        for k in range(common_set.shape[0]):

            cx_all_mmd = cx_all_mmd_all[i]
            cy_all_mmd = cy_all_mmd_all[i]
            x_obs_mmd =  x_obs_mmd_all[i]
            y_obs_mmd =  y_obs_mmd_all[i]

            cx_all_saa = cx_all_saa_all[i]
            cy_all_saa = cy_all_saa_all[i]
            x_obs_saa =  x_obs_saa_all[i]
            y_obs_saa =  y_obs_saa_all[i]
        
            cx_all_cvar = cx_all_cvar_all[i]
            cy_all_cvar = cy_all_cvar_all[i]
            x_obs_cvar =  x_obs_cvar_all[i]
            y_obs_cvar =  y_obs_cvar_all[i]
            
    for k in range(common_set.shape[0]):
        list_timestep=[0,68,99]
    
        th = np.linspace(0, 2*np.pi, 100)
        
        x_best = np.dot(prob0.P,cx_all_mmd[k].reshape(-1))
        y_best = np.dot(prob0.P,cy_all_mmd[k].reshape(-1))

        xdot_best = np.dot(prob0.Pdot,cx_all_mmd[k].reshape(-1))
        ydot_best = np.dot(prob0.Pdot,cy_all_mmd[k].reshape(-1))
        psi_best = np.arctan2(ydot_best,xdot_best)

        x_best_saa = np.dot(prob0.P,cx_all_saa[k].reshape(-1))
        y_best_saa = np.dot(prob0.P,cy_all_saa[k].reshape(-1))
        
        xdot_best_saa = np.dot(prob0.Pdot,cx_all_saa[k].reshape(-1))
        ydot_best_saa = np.dot(prob0.Pdot,cy_all_saa[k].reshape(-1))
        psi_best_saa = np.arctan2(ydot_best_saa,xdot_best_saa)

        x_best_cvar = np.dot(prob0.P,cx_all_cvar[k].reshape(-1))
        y_best_cvar = np.dot(prob0.P,cy_all_cvar[k].reshape(-1))
    
        xdot_best_cvar = np.dot(prob0.Pdot,cx_all_cvar[k].reshape(-1))
        ydot_best_cvar = np.dot(prob0.Pdot,cy_all_cvar[k].reshape(-1))
        psi_best_cvar = np.arctan2(ydot_best_cvar,xdot_best_cvar)

        lw_bound = 1.5
        lw_ego = 4
        lw_obs = 1
#########################################
        list_cost = args.costs
            
        #########################################

        fig,axs = plt.subplots(len(list_cost),len(list_timestep),figsize=(12,12),layout = "constrained")   

        ## X_obs - num_obs x batch x num
        _,X_obs,Y_obs,_ = compute_stats(k,cx_all_mmd[k],cy_all_mmd[k],
                                    x_obs_mmd[k],y_obs_mmd[k])
        
        for i,cost in enumerate(list_cost):
            for j,ts in enumerate(list_timestep):

                lb, = axs[i,j].plot(x_path_lb[0:len_path],y_path_lb[0:len_path],color='tab:brown',linewidth=3*linewidth,linestyle="--",
                        label="Lane boundary")
                
                axs[i,j].plot(x_path_ub[0:len_path],y_path_ub[0:len_path], color='tab:brown',linewidth=3*linewidth,linestyle="--")
                axs[i,j].plot(x_path_d_mid[0:len_path],y_path_d_mid[0:len_path], color='tab:brown',linewidth=3*linewidth,linestyle="--")

        ### Adding uncertainty trajectories

                for ii in range(num_obs):
                    _x_obs_noise_shift = X_obs[ii,:,0] - 2.5
                    _y_obs_noise_shift = Y_obs[ii,:,0] - 1.25 

                    patches_noise = [pt.Rectangle(center, width,height) for center, width,height in \
                    zip(np.vstack((_x_obs_noise_shift,_y_obs_noise_shift)).T, 5*np.ones(_num_batch),2.5*np.ones(_num_batch))]        
                    
                    coll = matplotlib.collections.PatchCollection(patches_noise,fc='red',alpha = 0.04)
                    axs[i,j].add_collection(coll)
                
                traj, = axs[i,j].plot(x_best[0],y_best[0],"g*",markersize = 7, label="Ego start position")

            ####### Ego patch           _
                if cost=="mmd_opt":
                            
                    ego_traj, = axs[i,j].plot(x_best[0:ts+1],y_best[0:ts+1]
                                            , "b", linewidth=3, label="Ego trajectory") 

                    
                    patch_ego = pt.Rectangle((x_best[ts]-2.5,y_best[ts]-1.25), 
                                            5,2.5,np.rad2deg(psi_best[ts]), fc = "orange",ec = "black",linewidth = 2,
                                                label="Ego")
                    axs[i,j].add_patch(patch_ego)
                    
                    _count,_,_,_ = compute_stats(k,cx_all_mmd[k],cy_all_mmd[k],
                                    x_obs_mmd[k],y_obs_mmd[k])
                    
                elif cost=="cvar":
                            
                    ego_traj, = axs[i,j].plot(x_best_cvar[0:ts+1],y_best_cvar[0:ts+1]
                                            , "b", linewidth=3, label="Ego trajectory")    
                    
                    patch_ego = pt.Rectangle((x_best_cvar[ts]-2.5,y_best_cvar[ts]-1.25), 
                                            5,2.5,np.rad2deg(psi_best_cvar[ts]), fc = "orange",ec = "black",linewidth = 2,
                                                label="Ego")
                    axs[i,j].add_patch(patch_ego)
                    
                    _count,_,_,_ = compute_stats(k,cx_all_cvar[k],cy_all_cvar[k],
                                    x_obs_cvar[k],y_obs_cvar[k])

                else:
                    ego_traj, = axs[i,j].plot(x_best_saa[0:ts+1],y_best_saa[0:ts+1]
                                            , "b", linewidth=3, label="Ego trajectory")    
                    
                    patch_ego = pt.Rectangle((x_best_saa[ts]-2.5,y_best_saa[ts]-1.25), 
                                            5,2.5,np.rad2deg(psi_best_saa[ts]), fc = "orange",ec = "black",linewidth = 2,
                                                label="Ego")
                    axs[i,j].add_patch(patch_ego)
                    
                    _count,_,_,_ = compute_stats(k,cx_all_saa[k],cy_all_saa[k],
                                    x_obs_saa[k],y_obs_saa[k])

                axs[i,j].set_ylabel('y [m]', fontweight = "bold", fontsize = fs)
                axs[i,j].set_xlabel('x [m]', fontweight = "bold", fontsize = fs)

                axs[i,j].set_aspect('equal', adjustable='box')

                axs[i,j].set(xlim=(-3, 115), ylim=(-20, 20))
                if j==2:
                    axs[i,j].text(0,17,'{}% collisions({}/{})'.format(round(100*_count/_num_batch,2),_count,_num_batch))
                
                loc = "center"
                if cost=="mmd_opt":
                    axs[i,j].set_title("$r_{MMD}^{emp}$, "+"timestep={}".format(ts), fontweight="bold",fontsize=fs,loc=loc)
                elif cost=="saa":
                    axs[i,j].set_title("$r_{SAA}$, "+"timestep={}".format(ts), fontweight="bold",fontsize=fs,loc=loc)
                else:
                    axs[i,j].set_title("$r_{CVaR}^{emp}$, "+"timestep={}".format(ts), fontweight="bold",fontsize=fs,loc=loc)

        plt.show()
