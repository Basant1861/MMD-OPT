from functools import partial
import sys
sys.path.insert(1, '/home/ims-robotics/MMD-OPT/autonomous_driving_synthetic_data/optimizer')
from kernel_computation import kernel_matrix
from compute_beta import beta_cem
from optimizer import cem
import numpy as np
import jax.numpy as jnp
from jax import lax,jit,vmap
import matplotlib.pyplot as plt
import jax
import scipy
import matplotlib.patches as pt
import matplotlib
import seaborn as sns
import argparse
from obs_data_generate_static import compute_obs_trajectories

sns.set_theme(style = "whitegrid", palette = 'tab10')
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('font', weight='bold')

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
       
        intersection = np.count_nonzero(cost,axis=1) #  num 
        count[tt] = np.max(intersection)

        X_obs1[tt], Y_obs1[tt], Psi_obs1[tt] = _x_obs_traj, _y_obs_traj, _psi_obs_traj
    

    return np.max(count),X_obs1,Y_obs1,Psi_obs1

_num_batch = 10**3 

parser = argparse.ArgumentParser()
parser.add_argument("--num_obs",  type=int, nargs='+', required=True)
parser.add_argument('-l','--num_reduced_set',type=int, nargs='+', required=True)
parser.add_argument('--noises',type=str, nargs='+', required=True)
parser.add_argument("--num_exps",  type=int, required=True)

args = parser.parse_args()
num_reduced_list = args.num_reduced_set
num_mother = _num_batch
list_num_obs = args.num_obs # number of nearest obs to take into account
num_exps = args.num_exps
noises = args.noises

for num_obs in list_num_obs:
    for noise in noises:
        if noise == "gaussian":
            from obs_data_generate_static import compute_noisy_init_state_gaussian as compute_noisy_init_state
        elif noise == "bimodal":
            from obs_data_generate_static import compute_noisy_init_state_bimodal as compute_noisy_init_state
        else:
            from obs_data_generate_static import compute_noisy_init_state_trimodal as compute_noisy_init_state

        root = "./data/static/{}".format(noise)

        for num_reduced in num_reduced_list:
            prob0 = cem.CEM(num_reduced,num_mother,num_obs,"static")

            dt = prob0.t
        
            key0 = prob0.key
            num = prob0.num_prime
            v_min = prob0.v_min
            v_max = prob0.v_max

            print("num_samples ", num_reduced)

            coll_mmd = []
            coll_saa = []
            coll_cvar = []
            coll_rand = []

            mmd_set_all,mmd_rand_set_all,cvar_set_all,saa_set_all = [],[],[],[]
            
            ## This block collects the common obstacle configs across all the experiments
            for l in range(num_exps):
                data_saa = np.load(root + "/saa_{}_samples_{}_obs_{}.npz".format(num_reduced,num_obs,l))
                data_cvar = np.load(root + "/cvar_{}_samples_{}_obs_{}.npz".format(num_reduced,num_obs,l))
                # data_rand = np.load(root + "/mmd_rand_{}_obs_{}_samples_{}.npz".format(num_obs,num_reduced,l))
                data = np.load(root + "/mmd_opt_{}_samples_{}_obs_{}.npz".format(num_reduced,num_obs,l))
            
                cx_all = np.asarray(data["cx"])
                cy_all = np.asarray(data["cy"])
                # init_state_all = np.asarray(data["init_state"])
                x_obs =  np.asarray(data["x_obs"])
                y_obs =  np.asarray(data["y_obs"])
                # psi_obs = np.asarray(data["psi_obs_all"])

                # cx_all_rand = np.asarray(data_rand["cx"])
                # cy_all_rand = np.asarray(data_rand["cy"])
                # init_state_all_rand = np.asarray(data_rand["init_state"])
                # x_obs_rand =  np.asarray(data_rand["x_obs"])
                # y_obs_rand =  np.asarray(data_rand["y_obs"])
                # psi_obs_rand = np.asarray(data_rand["psi_obs_all"])

                cx_all_saa = np.asarray(data_saa["cx"])
                cy_all_saa = np.asarray(data_saa["cy"])
                # init_state_all_saa = np.asarray(data_saa["init_state"])
                x_obs_saa =  np.asarray(data_saa["x_obs"])
                y_obs_saa =  np.asarray(data_saa["y_obs"])
                # psi_obs_saa = np.asarray(data_saa["psi_obs_all"])

                cx_all_cvar = np.asarray(data_cvar["cx"])
                cy_all_cvar = np.asarray(data_cvar["cy"])
                # init_state_all_cvar = np.asarray(data_cvar["init_state"])
                x_obs_cvar =  np.asarray(data_cvar["x_obs"])
                y_obs_cvar =  np.asarray(data_cvar["y_obs"])
                # psi_obs_cvar = np.asarray(data_cvar["psi_obs_all"])

                mmd_matrix = np.hstack((x_obs[:,0:num_obs],y_obs[:,0:num_obs]))
                mmd_set = set([tuple(x) for x in mmd_matrix])
                mmd_set_all.append(mmd_set)

                # mmd_rand_matrix = np.hstack((init_state_all_rand,x_obs_rand[:,0:num_obs],y_obs_rand[:,0:num_obs],psi_obs_rand[:,0:num_obs]))
                # mmd_rand_set = set([tuple(x) for x in mmd_rand_matrix])
                # mmd_rand_set_all.append(mmd_rand_set)

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
            
            # for x in mmd_rand_set_all[0]:
            #     count  = 0
            #     for pp in range(1,num_exps):
            #         if x in mmd_rand_set_all[pp]:
            #             count += 1
            #     if count == num_exps-1:
            #         x = np.array([x])
            #         mmd_rand_set_final = np.vstack((mmd_rand_set_final,x))

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

            # temp1 = set([tuple(x) for x in mmd_rand_set_final])
            temp2 = set([tuple(x) for x in saa_set_final])
            temp3 = set([tuple(x) for x in cvar_set_final])
            temp4 = set([tuple(x) for x in mmd_set_final])

            common_set = np.array([x for x in temp2 & temp3 & temp4])
            print(common_set.shape[0])

        ####### End of block #########################################
            
        ### The above block gives those obs configs which are present across all experiments;
        ### We still need to collect all data corresponding to these common obs configs across all experiments

            cx_all_mmd_all,cy_all_mmd_all,init_state_all_mmd_all,\
                x_obs_mmd_all,y_obs_mmd_all,psi_obs_mmd_all\
            = np.zeros((num_exps,common_set.shape[0],prob0.nvar)),\
                np.zeros((num_exps,common_set.shape[0],prob0.nvar)),\
                    np.zeros((num_exps,common_set.shape[0],6)),\
                        np.zeros((num_exps,common_set.shape[0],num_obs)),\
                            np.zeros((num_exps,common_set.shape[0],num_obs)),\
                                np.zeros((num_exps,common_set.shape[0],num_obs)) # num_exps x num_obs_configs x _

            cx_all_rand_all,cy_all_rand_all,init_state_all_rand_all,\
                x_obs_rand_all,y_obs_rand_all,psi_obs_rand_all\
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
                # data_rand = np.load(root + "/mmd_rand_{}_obs_{}_samples_{}.npz".format(num_obs,num_reduced,l))
                data = np.load(root + "/mmd_opt_{}_samples_{}_obs_{}.npz".format(num_reduced,num_obs,l))
            
                cx_all = np.asarray(data["cx"])
                cy_all = np.asarray(data["cy"])
                # init_state_all = np.asarray(data["init_state"])
                x_obs =  np.asarray(data["x_obs"])
                y_obs =  np.asarray(data["y_obs"])
                # psi_obs = np.asarray(data["psi_obs_all"])

                # cx_all_rand = np.asarray(data_rand["cx"])
                # cy_all_rand = np.asarray(data_rand["cy"])
                # init_state_all_rand = np.asarray(data_rand["init_state"])
                # x_obs_rand =  np.asarray(data_rand["x_obs"])
                # y_obs_rand =  np.asarray(data_rand["y_obs"])
                # psi_obs_rand = np.asarray(data_rand["psi_obs_all"])

                cx_all_saa = np.asarray(data_saa["cx"])
                cy_all_saa = np.asarray(data_saa["cy"])
                # init_state_all_saa = np.asarray(data_saa["init_state"])
                x_obs_saa =  np.asarray(data_saa["x_obs"])
                y_obs_saa =  np.asarray(data_saa["y_obs"])
                # psi_obs_saa = np.asarray(data_saa["psi_obs_all"])

                cx_all_cvar = np.asarray(data_cvar["cx"])
                cy_all_cvar = np.asarray(data_cvar["cy"])
                # init_state_all_cvar = np.asarray(data_cvar["init_state"])
                x_obs_cvar =  np.asarray(data_cvar["x_obs"])
                y_obs_cvar =  np.asarray(data_cvar["y_obs"])
                # psi_obs_cvar = np.asarray(data_cvar["psi_obs_all"])
                
                mmd_matrix = np.hstack((x_obs[:,0:num_obs],
                                y_obs[:,0:num_obs]))
                
                # mmd_rand_matrix = np.hstack((init_state_all_rand,
                #     x_obs_rand[:,0:num_obs],y_obs_rand[:,0:num_obs]
                #     ,psi_obs_rand[:,0:num_obs])) 
                
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
                    # init_state_all_mmd_all[l][k] = init_state_all[idx_mmd] 
                    # psi_obs_mmd_all[l][k] = psi_obs[idx_mmd] 
                
                    # idx_rand = np.where(np.all(common_set[k]==mmd_rand_matrix,axis=1))
                    # if len(idx_rand[0]) > 1 :
                    #     idx_rand = idx_rand[0][0]

                    # cx_all_rand_all[l][k] = cx_all_rand[idx_rand] 
                    # cy_all_rand_all[l][k] = cy_all_rand[idx_rand] 
                    # x_obs_rand_all[l][k] = x_obs_rand[idx_rand] 
                    # y_obs_rand_all[l][k] = y_obs_rand[idx_rand] 
                    # init_state_all_rand_all[l][k] = init_state_all_rand[idx_rand] 
                    # psi_obs_rand_all[l][k] = psi_obs_rand[idx_rand] 

                    idx_saa = np.where(np.all(common_set[k]== saa_matrix,axis=1))
                    if len(idx_saa[0]) > 1 :
                        idx_saa = idx_saa[0][0]

                    cx_all_saa_all[l][k] = cx_all_saa[idx_saa] 
                    cy_all_saa_all[l][k] = cy_all_saa[idx_saa] 
                    x_obs_saa_all[l][k] = x_obs_saa[idx_saa] 
                    y_obs_saa_all[l][k] = y_obs_saa[idx_saa] 
                    # init_state_all_saa_all[l][k] = init_state_all_saa[idx_saa] 
                    # psi_obs_saa_all[l][k] = psi_obs_saa[idx_saa] 

                    idx_cvar = np.where(np.all(common_set[k]==cvar_matrix,axis=1))
                    if len(idx_cvar[0]) > 1 :
                        idx_cvar = idx_cvar[0][0]

                    cx_all_cvar_all[l][k] = cx_all_cvar[idx_cvar] 
                    cy_all_cvar_all[l][k] = cy_all_cvar[idx_cvar] 
                    x_obs_cvar_all[l][k] = x_obs_cvar[idx_cvar] 
                    y_obs_cvar_all[l][k] = y_obs_cvar[idx_cvar] 
                    # init_state_all_cvar_all[l][k] = init_state_all_cvar[idx_cvar] 
                    # psi_obs_cvar_all[l][k] = psi_obs_cvar[idx_cvar] 

        ##### End of block #######
                    
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

            count_mmd_all,count_rand_all,count_saa_all,count_cvar_all = np.zeros((num_exps,common_set.shape[0])),\
                                                        np.zeros((num_exps,common_set.shape[0])),np.zeros((num_exps,common_set.shape[0])),\
                                                        np.zeros((num_exps,common_set.shape[0]))

            for i in range(num_exps):
                for k in range(common_set.shape[0]):

                    cx_all_mmd = cx_all_mmd_all[i]
                    cy_all_mmd = cy_all_mmd_all[i]
                    # init_state_all_mmd = init_state_all_mmd_all[i]
                    x_obs_mmd =  x_obs_mmd_all[i]
                    y_obs_mmd =  y_obs_mmd_all[i]
                    # psi_obs_mmd = psi_obs_mmd_all[i]
                
                    count_mmd,_,_,_ = compute_stats(k,cx_all_mmd[k],cy_all_mmd[k],
                                            x_obs_mmd[k],y_obs_mmd[k])
                
                    count_mmd_all[i][k] = count_mmd

                    # cx_all_rand = cx_all_rand_all[i]
                    # cy_all_rand = cy_all_rand_all[i]
                    # init_state_all_rand = init_state_all_rand_all[i]
                    # x_obs_rand =  x_obs_rand_all[i]
                    # y_obs_rand =  y_obs_rand_all[i]
                    # psi_obs_rand = psi_obs_rand_all[i]
                
                    # count_rand,_,_,_ = compute_stats(k,cx_all_rand[k],cy_all_rand[k],
                    #                         x_obs_rand[k],y_obs_rand[k],psi_obs_rand[k])
                    
                    # count_rand_all[i][k] = count_rand

                    cx_all_saa = cx_all_saa_all[i]
                    cy_all_saa = cy_all_saa_all[i]
                    # init_state_all_saa = init_state_all_saa_all[i]
                    x_obs_saa =  x_obs_saa_all[i]
                    y_obs_saa =  y_obs_saa_all[i]
                    # psi_obs_saa = psi_obs_saa_all[i]
                
                    count_saa,_,_,_ = compute_stats(k,cx_all_saa[k],cy_all_saa[k],
                                            x_obs_saa[k],y_obs_saa[k])
                    
                    count_saa_all[i][k] = count_saa

                    cx_all_cvar = cx_all_cvar_all[i]
                    cy_all_cvar = cy_all_cvar_all[i]
                    # init_state_all_cvar = init_state_all_cvar_all[i]
                    x_obs_cvar =  x_obs_cvar_all[i]
                    y_obs_cvar =  y_obs_cvar_all[i]
                    # psi_obs_cvar = psi_obs_cvar_all[i]
                
                    count_cvar,_,_,_ = compute_stats(k,cx_all_cvar[k],cy_all_cvar[k],
                                            x_obs_cvar[k],y_obs_cvar[k])
                    
                    count_cvar_all[i][k] = count_cvar

            coll_rand = np.mean(count_rand_all,axis=0)
            coll_saa = np.mean(count_saa_all,axis=0)
            coll_cvar = np.mean(count_cvar_all,axis=0)
            coll_mmd = np.mean(count_mmd_all,axis=0)

        #########################################
                
            np.savez("./stats/static/{}/stats_{}_samples_{}_obs".format(noise,num_reduced,num_obs), 
                    coll_mmd = coll_mmd,coll_saa=coll_saa,
                    coll_cvar = coll_cvar,coll_rand = coll_rand)

