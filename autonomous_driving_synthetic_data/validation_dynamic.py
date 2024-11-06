import sys
sys.path.insert(1, '/home/ims-robotics/MMD-OPT/autonomous_driving_synthetic_data/optimizer')
from optimizer import cem
import numpy as np
import jax.numpy as jnp
from jax import jit
import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib.patches as pt
import matplotlib
import seaborn as sns
import argparse
import bernstein_coeff_order10_arbitinterval
import jax.numpy as jnp
import jax
from jax import jit

sns.set_theme(style = "whitegrid", palette = 'tab10')
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('font', weight='bold')

a_min,a_max,steer_min,steer_max = -18.,18.,-0.6,0.6

@jit	
def compute_boundary_vec(x_init, vx_init, ax_init, y_init, vy_init, ay_init):

    x_init_vec = x_init*jnp.ones((_num_batch, 1))
    y_init_vec = y_init*jnp.ones((_num_batch, 1))

    vx_init_vec = vx_init*jnp.ones((_num_batch, 1))
    vy_init_vec = vy_init*jnp.ones((_num_batch, 1))

    ax_init_vec = ax_init*jnp.ones((_num_batch, 1))
    ay_init_vec = ay_init*jnp.ones((_num_batch, 1))

    b_eq_x = jnp.hstack(( x_init_vec, vx_init_vec, ax_init_vec ))
    b_eq_y = jnp.hstack(( y_init_vec, vy_init_vec, ay_init_vec, jnp.zeros((_num_batch, 1  ))   ))

    return b_eq_x, b_eq_y

@jit	
def compute_obs_guess(b_eq_x,b_eq_y,y_samples,seed):

    v_des = sampling_param(seed)
    y_des = y_samples

    #############################
    A_vd = Pddot-k_p_v*Pdot
    b_vd = -k_p_v*jnp.ones((_num_batch, num))*(v_des)[:, jnp.newaxis]
    
    A_pd = Pddot-k_p*P
    b_pd = -k_p*jnp.ones((_num_batch, num ))*(y_des)[:, jnp.newaxis]

    cost_smoothness_x = weight_smoothness_x*jnp.dot(Pddot.T, Pddot)
    cost_smoothness_y = weight_smoothness_y*jnp.dot(Pddot.T, Pddot)
    
    cost_x = cost_smoothness_x+rho_v*jnp.dot(A_vd.T, A_vd)
    cost_y = cost_smoothness_y+rho_offset*jnp.dot(A_pd.T, A_pd)

    cost_mat_x = jnp.vstack((  jnp.hstack(( cost_x, A_eq_x.T )), jnp.hstack(( A_eq_x, jnp.zeros(( jnp.shape(A_eq_x)[0], jnp.shape(A_eq_x)[0] )) )) ))
    cost_mat_y = jnp.vstack((  jnp.hstack(( cost_y, A_eq_y.T )), jnp.hstack(( A_eq_y, jnp.zeros(( jnp.shape(A_eq_y)[0], jnp.shape(A_eq_y)[0] )) )) ))
    
    lincost_x = -rho_v*jnp.dot(A_vd.T, b_vd.T).T
    lincost_y = -rho_offset*jnp.dot(A_pd.T, b_pd.T).T

    sol_x = jnp.linalg.solve(cost_mat_x, jnp.hstack(( -lincost_x, b_eq_x )).T).T
    sol_y = jnp.linalg.solve(cost_mat_y, jnp.hstack(( -lincost_y, b_eq_y )).T).T

    #######################

    primal_sol_x = sol_x[:,0:nvar]
    primal_sol_y = sol_y[:,0:nvar]

    x = jnp.dot(P, primal_sol_x.T).T
    y = jnp.dot(P, primal_sol_y.T).T
    
    return x,y

@jit	
def sampling_param(seed):
    
    param_samples = jax.random.normal(jax.random.PRNGKey(seed),(_num_batch,))

    eps_v1 = param_samples*v_sigma[0] + v_mu[0]
    eps_v2 = param_samples*v_sigma[1] + v_mu[1]
    eps_v3 = param_samples*v_sigma[2] + v_mu[2]

    eps_v1 = jnp.clip(eps_v1, v_min*jnp.ones(_num_batch),v_max*jnp.ones(_num_batch)   )
    eps_v2 = jnp.clip(eps_v2, v_min*jnp.ones(_num_batch),v_max*jnp.ones(_num_batch)   )
    eps_v3 = jnp.clip(eps_v3, v_min*jnp.ones(_num_batch),v_max*jnp.ones(_num_batch)   )

    weight_samples = jax.random.choice(jax.random.PRNGKey(seed),modes, (_num_batch,),p=modes_probs)

    idx_1 = jnp.where(weight_samples==1,size=size_1)
    idx_2 = jnp.where(weight_samples==2,size=size_2)
    idx_3 = jnp.where(weight_samples==3,size=size_3)

    eps_v = jnp.hstack((eps_v1[idx_1],eps_v2[idx_2],eps_v3[idx_3]))

    return eps_v

def normal_vectors(x, y, scalar):
    tck = scipy.interpolate.splrep(x, y)
    y_deriv = scipy.interpolate.splev(x, tck, der=1)
    normals_rad = np.arctan(y_deriv)+np.pi/2.
    return np.cos(normals_rad)*scalar, np.sin(normals_rad)*scalar

@jit
def compute_f_bar_temp(x_obs,y_obs,x,y): 
    
    wc_alpha = (x-x_obs[:,0:num])
    ws_alpha = (y-y_obs[:,0:num])

    cost = -(wc_alpha**2)/(prob0.a_obs**2) - (ws_alpha**2)/(prob0.b_obs**2) + jnp.ones((_num_batch,num))

    costbar = jnp.maximum(jnp.zeros((_num_batch,num)),cost)
        
    return costbar 

@jit
def compute_stats(seed,_y_des_1,_y_des_2,prob_des,cx,cy,x_obs,y_obs,vx_obs,vy_obs):

    prob_des = prob_des.reshape(-1)
    _y_des_1 = _y_des_1.reshape(-1)
    _y_des_2 = _y_des_2.reshape(-1)

    x_obs = x_obs.reshape(-1)
    y_obs = y_obs.reshape(-1)
    vx_obs = vx_obs.reshape(-1)
    vy_obs = vy_obs.reshape(-1)
    
    X_obs1,Y_obs1 =  jnp.zeros((num_obs,_num_batch,num)),\
jnp.zeros((num_obs,_num_batch,num))
   
    x = jnp.dot(jnp.asarray(P),cx.reshape(-1))
    y = jnp.dot(jnp.asarray(P),cy.reshape(-1))

    count = jnp.zeros(prob0.num_obs)
    print(num_obs)

    for tt in range(num_obs):
        b_eq_x,b_eq_y = compute_boundary_vec(x_obs[tt],vx_obs[tt],0.,y_obs[tt],vy_obs[tt],0.)

        y_des= jnp.asarray([_y_des_1[tt],_y_des_2[tt]]).reshape(-1)
        probabilities = jnp.asarray([prob_des[tt],1-prob_des[tt]]).reshape(-1)
       
        y_samples = jax.random.choice(jax.random.PRNGKey(seed + tt),y_des, (_num_batch,), p=probabilities)
        _x_obs_traj,_y_obs_traj = compute_obs_guess(b_eq_x,b_eq_y,y_samples,seed)

        cost = compute_f_bar_temp(_x_obs_traj,_y_obs_traj,x,y) # _num_batch x num
        cost = cost.T # num x _num_batch

        intersection = jnp.count_nonzero(cost,axis=1) # num
        count = count.at[tt].set(jnp.max(intersection))

        X_obs1 = X_obs1.at[tt].set(_x_obs_traj)
        Y_obs1 = Y_obs1.at[tt].set(_y_obs_traj)
        
    return jnp.max(count),X_obs1,Y_obs1

_num_batch = 10**3 
t_fin = 15
num = 100
t = t_fin/num
tot_time = np.linspace(0, t_fin, num)
tot_time = tot_time
tot_time_copy = tot_time.reshape(num, 1)

v_min,v_max = 0.1,30.
P, Pdot, Pddot = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)

nvar = np.shape(P)[1]

weight_smoothness_x = 100
weight_smoothness_y = 100

rho_v = 1 
rho_offset = 1

k_p_v = 2
k_d_v = 2.0*np.sqrt(k_p_v)

k_p = 2
k_d = 2.0*np.sqrt(k_p)

A_eq_x = np.vstack(( P[0], Pdot[0], Pddot[0]  ))
A_eq_y = np.vstack(( P[0], Pdot[0], Pddot[0], Pdot[-1]  ))

modes = np.asarray([1,2,3])
modes_probs = np.asarray([0.4,0.2,0.4])
v_mu = np.asarray([5.,7.,3.])
v_sigma = np.asarray([1.5,0.1,3]) 
size_1 = jnp.array(modes_probs[0]*_num_batch,int) 
size_2 = jnp.array(modes_probs[1]*_num_batch,int) 
size_3 = jnp.array(modes_probs[2]*_num_batch,int) 

size_1 = jax.lax.cond(size_1+size_2+size_3==_num_batch,
                        lambda _: size_1,lambda _: _num_batch-(size_2 + size_3), 0)

parser = argparse.ArgumentParser()
parser.add_argument("--num_obs",  type=int, nargs='+', required=True)
parser.add_argument('-l','--num_reduced_set',type=int, nargs='+', required=True)
parser.add_argument("--num_exps",  type=int, required=True)
parser.add_argument('--setting',type=str, nargs='+', required=True)

args = parser.parse_args()
num_reduced_list = args.num_reduced_set
num_mother = _num_batch
list_num_obs = args.num_obs # number of nearest obs to take into account
num_exps = args.num_exps
list_setting = args.setting

for num_obs in list_num_obs:
    for sc in list_setting:
        root = "./data/dynamic/{}".format(sc)

        for num_reduced in num_reduced_list:
            prob0 = cem.CEM(num_reduced,num_mother,num_obs,sc)    
                
            print("num_samples ", num_reduced,"num_obs", num_obs,"setting", sc)

            coll_mmd = []
            coll_saa = []
            coll_cvar = []
            coll_rand = []

            vel_mmd = []
            vel_saa = []
            vel_cvar = []
            vel_rand = []

            mmd_set_all,mmd_rand_set_all,cvar_set_all,saa_set_all = [],[],[],[]
            
            ## This block collects the common obstacle configs across all the experiments
            for l in range(num_exps):
                data_saa = np.load(root + "/saa_{}_samples_{}_obs_{}.npz".format(num_reduced,num_obs,l))
                data_cvar = np.load(root + "/cvar_{}_samples_{}_obs_{}.npz".format(num_reduced,num_obs,l))
                # data_rand = np.load(root + "/mmd_rand_{}_obs_{}_samples_{}.npz".format(num_obs,num_reduced,l))
                data = np.load(root + "/mmd_opt_{}_samples_{}_obs_{}.npz".format(num_reduced,num_obs,l))
                
                cx_all = np.asarray(data["cx"])
                cy_all = np.asarray(data["cy"])
                init_state_all = np.asarray(data["init_state"])
                x_obs =  np.asarray(data["x_obs_all"])
                y_obs =  np.asarray(data["y_obs_all"])
                vx_obs =  np.asarray(data["vx_obs_all"])
                vy_obs =  np.asarray(data["vy_obs_all"])
                psi_obs = np.asarray(data["psi_obs_all"])
                prob_obs = np.asarray(data["prob"])
                y_des_1 = np.asarray(data["y_des_1"])
                y_des_2 = np.asarray(data["y_des_2"])

                # cx_all_rand = np.asarray(data_rand["cx"])
                # cy_all_rand = np.asarray(data_rand["cy"])
                # init_state_all_rand = np.asarray(data_rand["init_state"])
                # x_obs_rand =  np.asarray(data_rand["x_obs_all"])
                # y_obs_rand =  np.asarray(data_rand["y_obs_all"])
                # psi_obs_rand = np.asarray(data_rand["psi_obs_all"])
                # vx_obs_rand =  np.asarray(data_rand["vx_obs_all"])
                # vy_obs_rand =  np.asarray(data_rand["vy_obs_all"])
                # prob_obs_rand = np.asarray(data_rand["prob"])
                # x_obs_rand =  np.asarray(data_rand["x_obs_all"])
                # y_obs_rand =  np.asarray(data_rand["y_obs_all"])
                # y_des_1_rand = np.asarray(data_rand["y_des_1"])
                # y_des_2_rand = np.asarray(data_rand["y_des_2"])

                cx_all_saa = np.asarray(data_saa["cx"])
                cy_all_saa = np.asarray(data_saa["cy"])
                init_state_all_saa = np.asarray(data_saa["init_state"])
                x_obs_saa =  np.asarray(data_saa["x_obs_all"])
                y_obs_saa =  np.asarray(data_saa["y_obs_all"])
                vx_obs_saa =  np.asarray(data_saa["vx_obs_all"])
                vy_obs_saa =  np.asarray(data_saa["vy_obs_all"])
                psi_obs_saa = np.asarray(data_saa["psi_obs_all"])
                prob_obs_saa = np.asarray(data_saa["prob"])
                y_des_1_saa = np.asarray(data_saa["y_des_1"])
                y_des_2_saa = np.asarray(data_saa["y_des_2"])

                cx_all_cvar = np.asarray(data_cvar["cx"])
                cy_all_cvar = np.asarray(data_cvar["cy"])
                init_state_all_cvar = np.asarray(data_cvar["init_state"])
                x_obs_cvar =  np.asarray(data_cvar["x_obs_all"])
                y_obs_cvar =  np.asarray(data_cvar["y_obs_all"])
                vx_obs_cvar =  np.asarray(data_cvar["vx_obs_all"])
                vy_obs_cvar =  np.asarray(data_cvar["vy_obs_all"])
                psi_obs_cvar = np.asarray(data_cvar["psi_obs_all"])
                prob_obs_cvar = np.asarray(data_cvar["prob"])
                y_des_1_cvar = np.asarray(data_cvar["y_des_1"])
                y_des_2_cvar = np.asarray(data_cvar["y_des_2"])
                
                # print(init_state_all.shape,x_obs.shape,y_obs.shape,vx_obs.shape,vy_obs.shape,prob_obs.shape,y_des_1.shape,y_des_2.shape)
                
                mmd_matrix = np.hstack((init_state_all,x_obs[:,0:num_obs],y_obs[:,0:num_obs],
                                vx_obs[:,0:num_obs],vy_obs[:,0:num_obs],
                                prob_obs[:,0:num_obs],
                            y_des_1[:,0:num_obs], y_des_2[:,0:num_obs]))
                
                mmd_set = set([tuple(x) for x in mmd_matrix])
                mmd_set_all.append(mmd_set)

                # mmd_rand_matrix = np.hstack((init_state_all_rand,x_obs_rand[:,0:num_obs],y_obs_rand[:,0:num_obs],
                #                  vx_obs_rand[:,0:num_obs],vy_obs_rand[:,0:num_obs],prob_obs_rand.reshape(-1,1),
                #                     y_des_1_rand[:,0:num_obs], y_des_2_rand[:,0:num_obs]))
                
                # mmd_rand_set = set([tuple(x) for x in mmd_rand_matrix])
                # mmd_rand_set_all.append(mmd_rand_set)

                saa_matrix = np.hstack((init_state_all_saa,x_obs_saa[:,0:num_obs],y_obs_saa[:,0:num_obs],
                                vx_obs_saa[:,0:num_obs],vy_obs_saa[:,0:num_obs],prob_obs_saa[:,0:num_obs],
                                y_des_1_saa[:,0:num_obs], y_des_2_saa[:,0:num_obs]))
                
                saa_set = set([tuple(x) for x in saa_matrix])
                saa_set_all.append(saa_set)

                cvar_matrix = np.hstack((init_state_all_cvar,x_obs_cvar[:,0:num_obs],y_obs_cvar[:,0:num_obs],
                                vx_obs_cvar[:,0:num_obs],vy_obs_cvar[:,0:num_obs],prob_obs_cvar[:,0:num_obs],
                                y_des_1_cvar[:,0:num_obs], y_des_2_cvar[:,0:num_obs]))
                
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

            # print(mmd_rand_set_final.shape, mmd_set_final.shape,cvar_set_final.shape,saa_set_final.shape)
            # kk
            common_set = np.array([x for x in temp2 & temp3 & temp4])
            # common_set = np.array([x for x in temp4])

            # print(common_set.shape[0])

        ####### End of block #########################################
            
        ### The above block gives those obs configs which are present across all experiments;
        ### We still need to collect all data corresponding to these common obs configs across all experiments

            cx_all_mmd_all,cy_all_mmd_all,init_state_all_mmd_all,\
                x_obs_mmd_all,y_obs_mmd_all,psi_obs_mmd_all,\
                    vx_obs_mmd_all,vy_obs_mmd_all,prob_obs_mmd_all\
            = np.zeros((num_exps,common_set.shape[0],prob0.nvar)),\
                np.zeros((num_exps,common_set.shape[0],prob0.nvar)),\
                    np.zeros((num_exps,common_set.shape[0],6)),\
                        np.zeros((num_exps,common_set.shape[0],num_obs)),\
                            np.zeros((num_exps,common_set.shape[0],num_obs)),\
                                np.zeros((num_exps,common_set.shape[0],num_obs)),\
                                    np.zeros((num_exps,common_set.shape[0],num_obs)),\
                                        np.zeros((num_exps,common_set.shape[0],num_obs)),\
                                            np.zeros((num_exps,common_set.shape[0],num_obs)) # num_exps x num_obs_configs x _
            
            cx_all_rand_all,cy_all_rand_all,init_state_all_rand_all,\
                x_obs_rand_all,y_obs_rand_all,psi_obs_rand_all,\
                    vx_obs_rand_all,vy_obs_rand_all,prob_obs_rand_all\
            = np.zeros((num_exps,common_set.shape[0],prob0.nvar)),\
                np.zeros((num_exps,common_set.shape[0],prob0.nvar)),\
                    np.zeros((num_exps,common_set.shape[0],6)),\
                        np.zeros((num_exps,common_set.shape[0],num_obs)),\
                            np.zeros((num_exps,common_set.shape[0],num_obs)),\
                                np.zeros((num_exps,common_set.shape[0],num_obs)),\
                                    np.zeros((num_exps,common_set.shape[0],num_obs)),\
                                        np.zeros((num_exps,common_set.shape[0],num_obs)),\
                                            np.zeros((num_exps,common_set.shape[0],num_obs)) # num_exps x num_obs_configs x _

            cx_all_saa_all,cy_all_saa_all,init_state_all_saa_all,\
                x_obs_saa_all,y_obs_saa_all,psi_obs_saa_all,\
                vx_obs_saa_all,vy_obs_saa_all,prob_obs_saa_all\
            = np.zeros((num_exps,common_set.shape[0],prob0.nvar)),\
                np.zeros((num_exps,common_set.shape[0],prob0.nvar)),\
                    np.zeros((num_exps,common_set.shape[0],6)),\
                        np.zeros((num_exps,common_set.shape[0],num_obs)),\
                            np.zeros((num_exps,common_set.shape[0],num_obs)),\
                                np.zeros((num_exps,common_set.shape[0],num_obs)),\
                                    np.zeros((num_exps,common_set.shape[0],num_obs)),\
                                        np.zeros((num_exps,common_set.shape[0],num_obs)),\
                                            np.zeros((num_exps,common_set.shape[0],num_obs)) # num_exps x num_obs_configs x _

            cx_all_cvar_all,cy_all_cvar_all,init_state_all_cvar_all,\
                x_obs_cvar_all,y_obs_cvar_all,psi_obs_cvar_all,\
                        vx_obs_cvar_all,vy_obs_cvar_all,prob_obs_cvar_all\
            = np.zeros((num_exps,common_set.shape[0],prob0.nvar)),\
                np.zeros((num_exps,common_set.shape[0],prob0.nvar)),\
                    np.zeros((num_exps,common_set.shape[0],6)),\
                        np.zeros((num_exps,common_set.shape[0],num_obs)),\
                            np.zeros((num_exps,common_set.shape[0],num_obs)),\
                                np.zeros((num_exps,common_set.shape[0],num_obs)),\
                                    np.zeros((num_exps,common_set.shape[0],num_obs)),\
                                        np.zeros((num_exps,common_set.shape[0],num_obs)),\
                                            np.zeros((num_exps,common_set.shape[0],num_obs)) # num_exps x num_obs_configs x _
            
            y_des_1_rand_all,y_des_2_rand_all = np.zeros((num_exps,common_set.shape[0],num_obs)),\
                                                np.zeros((num_exps,common_set.shape[0],num_obs))
            
            y_des_1_saa_all,y_des_2_saa_all = np.zeros((num_exps,common_set.shape[0],num_obs)),\
                                                np.zeros((num_exps,common_set.shape[0],num_obs))
            
            y_des_1_cvar_all,y_des_2_cvar_all = np.zeros((num_exps,common_set.shape[0],num_obs)),\
                                                np.zeros((num_exps,common_set.shape[0],num_obs))
            
            y_des_1_mmd_all,y_des_2_mmd_all = np.zeros((num_exps,common_set.shape[0],num_obs)),\
                                                np.zeros((num_exps,common_set.shape[0],num_obs))
            
            for l in range(num_exps):
                data_saa = np.load(root + "/saa_{}_samples_{}_obs_{}.npz".format(num_reduced,num_obs,l))
                data_cvar = np.load(root + "/cvar_{}_samples_{}_obs_{}.npz".format(num_reduced,num_obs,l))
                # data_rand = np.load(root + "/mmd_rand_{}_obs_{}_samples_{}.npz".format(num_obs,num_reduced,l))
                data = np.load(root + "/mmd_opt_{}_samples_{}_obs_{}.npz".format(num_reduced,num_obs,l))
                
                cx_all = np.asarray(data["cx"])
                cy_all = np.asarray(data["cy"])
                init_state_all = np.asarray(data["init_state"])
                x_obs =  np.asarray(data["x_obs_all"])
                y_obs =  np.asarray(data["y_obs_all"])
                vx_obs =  np.asarray(data["vx_obs_all"])
                vy_obs =  np.asarray(data["vy_obs_all"])
                psi_obs = np.asarray(data["psi_obs_all"])
                prob_obs = np.asarray(data["prob"])
                y_des_1 = np.asarray(data["y_des_1"])
                y_des_2 = np.asarray(data["y_des_2"])

                # cx_all_rand = np.asarray(data_rand["cx"])
                # cy_all_rand = np.asarray(data_rand["cy"])
                # init_state_all_rand = np.asarray(data_rand["init_state"])
                # x_obs_rand =  np.asarray(data_rand["x_obs_all"])
                # y_obs_rand =  np.asarray(data_rand["y_obs_all"])
                # psi_obs_rand = np.asarray(data_rand["psi_obs_all"])
                # vx_obs_rand =  np.asarray(data_rand["vx_obs_all"])
                # vy_obs_rand =  np.asarray(data_rand["vy_obs_all"])
                # prob_obs_rand = np.asarray(data_rand["prob"])
                # x_obs_rand =  np.asarray(data_rand["x_obs_all"])
                # y_obs_rand =  np.asarray(data_rand["y_obs_all"])
                # y_des_1_rand = np.asarray(data_rand["y_des_1"])
                # y_des_2_rand = np.asarray(data_rand["y_des_2"])

                cx_all_saa = np.asarray(data_saa["cx"])
                cy_all_saa = np.asarray(data_saa["cy"])
                init_state_all_saa = np.asarray(data_saa["init_state"])
                x_obs_saa =  np.asarray(data_saa["x_obs_all"])
                y_obs_saa =  np.asarray(data_saa["y_obs_all"])
                vx_obs_saa =  np.asarray(data_saa["vx_obs_all"])
                vy_obs_saa =  np.asarray(data_saa["vy_obs_all"])
                psi_obs_saa = np.asarray(data_saa["psi_obs_all"])
                prob_obs_saa = np.asarray(data_saa["prob"])
                y_des_1_saa = np.asarray(data_saa["y_des_1"])
                y_des_2_saa = np.asarray(data_saa["y_des_2"])

                cx_all_cvar = np.asarray(data_cvar["cx"])
                cy_all_cvar = np.asarray(data_cvar["cy"])
                init_state_all_cvar = np.asarray(data_cvar["init_state"])
                x_obs_cvar =  np.asarray(data_cvar["x_obs_all"])
                y_obs_cvar =  np.asarray(data_cvar["y_obs_all"])
                vx_obs_cvar =  np.asarray(data_cvar["vx_obs_all"])
                vy_obs_cvar =  np.asarray(data_cvar["vy_obs_all"])
                psi_obs_cvar = np.asarray(data_cvar["psi_obs_all"])
                prob_obs_cvar = np.asarray(data_cvar["prob"])
                y_des_1_cvar = np.asarray(data_cvar["y_des_1"])
                y_des_2_cvar = np.asarray(data_cvar["y_des_2"])
                
                mmd_matrix = np.hstack((init_state_all,x_obs[:,0:num_obs],y_obs[:,0:num_obs],
                                vx_obs[:,0:num_obs],vy_obs[:,0:num_obs],
                                prob_obs[:,0:num_obs],
                            y_des_1[:,0:num_obs], y_des_2[:,0:num_obs]))
                
                # mmd_rand_matrix = np.hstack((init_state_all_rand,x_obs_rand[:,0:num_obs],y_obs_rand[:,0:num_obs],
                #                  vx_obs_rand[:,0:num_obs],vy_obs_rand[:,0:num_obs],prob_obs_rand.reshape(-1,1),
                #                     y_des_1_rand[:,0:num_obs], y_des_2_rand[:,0:num_obs]))
                
                saa_matrix = np.hstack((init_state_all_saa,x_obs_saa[:,0:num_obs],y_obs_saa[:,0:num_obs],
                                vx_obs_saa[:,0:num_obs],vy_obs_saa[:,0:num_obs],prob_obs_saa[:,0:num_obs],
                                y_des_1_saa[:,0:num_obs], y_des_2_saa[:,0:num_obs]))
            
                cvar_matrix = np.hstack((init_state_all_cvar,x_obs_cvar[:,0:num_obs],y_obs_cvar[:,0:num_obs],
                                vx_obs_cvar[:,0:num_obs],vy_obs_cvar[:,0:num_obs],prob_obs_cvar[:,0:num_obs],
                                y_des_1_cvar[:,0:num_obs], y_des_2_cvar[:,0:num_obs]))
                
                for k in range(common_set.shape[0]):

                    idx_mmd = np.where(np.all(common_set[k]==mmd_matrix,axis=1))
                    if len(idx_mmd[0]) > 1 :
                        idx_mmd = idx_mmd[0][0]

                    cx_all_mmd_all[l][k] = cx_all[idx_mmd] 
                    cy_all_mmd_all[l][k] = cy_all[idx_mmd] 
                    x_obs_mmd_all[l][k] = x_obs[idx_mmd] 
                    y_obs_mmd_all[l][k] = y_obs[idx_mmd] 
                    init_state_all_mmd_all[l][k] = init_state_all[idx_mmd] 
                    psi_obs_mmd_all[l][k] = psi_obs[idx_mmd] 
                    vx_obs_mmd_all[l][k] = vx_obs[idx_mmd] 
                    vy_obs_mmd_all[l][k] = vy_obs[idx_mmd] 
                    prob_obs_mmd_all[l][k] = prob_obs[idx_mmd]
                    y_des_1_mmd_all[l][k] = y_des_1[idx_mmd]
                    y_des_2_mmd_all[l][k] = y_des_2[idx_mmd]

                    # idx_rand = np.where(np.all(common_set[k]==mmd_rand_matrix,axis=1))
                    # if len(idx_rand[0]) > 1 :
                    #     idx_rand = idx_rand[0][0]

                    # cx_all_rand_all[l][k] = cx_all_rand[idx_rand] 
                    # cy_all_rand_all[l][k] = cy_all_rand[idx_rand] 
                    # x_obs_rand_all[l][k] = x_obs_rand[idx_rand] 
                    # y_obs_rand_all[l][k] = y_obs_rand[idx_rand] 
                    # init_state_all_rand_all[l][k] = init_state_all_rand[idx_rand] 
                    # psi_obs_rand_all[l][k] = psi_obs_rand[idx_rand] 
                    # vx_obs_rand_all[l][k] = vx_obs_rand[idx_rand] 
                    # vy_obs_rand_all[l][k] = vy_obs_rand[idx_rand] 
                    # prob_obs_rand_all[l][k] = prob_obs_rand[idx_rand] 
                    # y_des_1_rand_all[l][k] = y_des_1_rand[idx_rand]
                    # y_des_2_rand_all[l][k] = y_des_2_rand[idx_rand]

                    idx_saa = np.where(np.all(common_set[k]== saa_matrix,axis=1))
                    if len(idx_saa[0]) > 1 :
                        idx_saa = idx_saa[0][0]

                    cx_all_saa_all[l][k] = cx_all_saa[idx_saa] 
                    cy_all_saa_all[l][k] = cy_all_saa[idx_saa] 
                    x_obs_saa_all[l][k] = x_obs_saa[idx_saa] 
                    y_obs_saa_all[l][k] = y_obs_saa[idx_saa] 
                    init_state_all_saa_all[l][k] = init_state_all_saa[idx_saa] 
                    psi_obs_saa_all[l][k] = psi_obs_saa[idx_saa] 
                    vx_obs_saa_all[l][k] = vx_obs_saa[idx_saa] 
                    vy_obs_saa_all[l][k] = vy_obs_saa[idx_saa] 
                    prob_obs_saa_all[l][k] = prob_obs_saa[idx_saa] 
                    y_des_1_saa_all[l][k] = y_des_1_saa[idx_saa]
                    y_des_2_saa_all[l][k] = y_des_2_saa[idx_saa]

                    idx_cvar = np.where(np.all(common_set[k]==cvar_matrix,axis=1))
                    if len(idx_cvar[0]) > 1 :
                        idx_cvar = idx_cvar[0][0]

                    cx_all_cvar_all[l][k] = cx_all_cvar[idx_cvar] 
                    cy_all_cvar_all[l][k] = cy_all_cvar[idx_cvar] 
                    x_obs_cvar_all[l][k] = x_obs_cvar[idx_cvar] 
                    y_obs_cvar_all[l][k] = y_obs_cvar[idx_cvar] 
                    init_state_all_cvar_all[l][k] = init_state_all_cvar[idx_cvar] 
                    psi_obs_cvar_all[l][k] = psi_obs_cvar[idx_cvar] 
                    vx_obs_cvar_all[l][k] = vx_obs_cvar[idx_cvar] 
                    vy_obs_cvar_all[l][k] = vy_obs_cvar[idx_cvar] 
                    prob_obs_cvar_all[l][k] = prob_obs_cvar[idx_cvar] 
                    y_des_1_cvar_all[l][k] = y_des_1_cvar[idx_cvar]
                    y_des_2_cvar_all[l][k] = y_des_2_cvar[idx_cvar]

        ##### End of block #######
          
            count_mmd_all,count_rand_all,count_saa_all,count_cvar_all = np.zeros((num_exps,common_set.shape[0])),\
                                                        np.zeros((num_exps,common_set.shape[0])),np.zeros((num_exps,common_set.shape[0])),\
                                                        np.zeros((num_exps,common_set.shape[0]))

            for i in range(num_exps):
                for k in range(common_set.shape[0]):

                    cx_all_mmd = cx_all_mmd_all[i]
                    cy_all_mmd = cy_all_mmd_all[i]
                    init_state_all_mmd = init_state_all_mmd_all[i]
                    x_obs_mmd =  x_obs_mmd_all[i]
                    y_obs_mmd =  y_obs_mmd_all[i]
                    psi_obs_mmd = psi_obs_mmd_all[i]
                    vx_obs_mmd =  vx_obs_mmd_all[i]
                    vy_obs_mmd =  vy_obs_mmd_all[i]
                    prob_obs_mmd = prob_obs_mmd_all[i]
                    y_des_1_mmd = y_des_1_mmd_all[i]
                    y_des_2_mmd = y_des_2_mmd_all[i]

                    count_mmd,_,_ = compute_stats(13*i+ 5*k + 17,y_des_1_mmd[k],y_des_2_mmd[k],
                                                prob_obs_mmd[k],cx_all_mmd[k],cy_all_mmd[k],
                                            x_obs_mmd[k],y_obs_mmd[k],vx_obs_mmd[k],vy_obs_mmd[k])
                
                    count_mmd_all[i][k] = count_mmd

                    # cx_all_rand = cx_all_rand_all[i]
                    # cy_all_rand = cy_all_rand_all[i]
                    # init_state_all_rand = init_state_all_rand_all[i]
                    # x_obs_rand =  x_obs_rand_all[i]
                    # y_obs_rand =  y_obs_rand_all[i]
                    # psi_obs_rand = psi_obs_rand_all[i]
                    # vx_obs_rand =  vx_obs_rand_all[i]
                    # vy_obs_rand =  vy_obs_rand_all[i]
                    # prob_obs_rand = prob_obs_rand_all[i]
                    # y_des_1_rand = y_des_1_rand_all[i]
                    # y_des_2_rand = y_des_2_rand_all[i]

                    # count_rand,_,_ = compute_stats(13*i+ 5*k + 17,y_des_1_rand[k],y_des_2_rand[k],
                    #                                prob_obs_rand[k],cx_all_rand[k],cy_all_rand[k],
                    #                         x_obs_rand[k],y_obs_rand[k],vx_obs_rand[k],vy_obs_rand[k])
                
                    # count_rand_all[i][k] = count_rand

                    cx_all_saa = cx_all_saa_all[i]
                    cy_all_saa = cy_all_saa_all[i]
                    init_state_all_saa = init_state_all_saa_all[i]
                    x_obs_saa =  x_obs_saa_all[i]
                    y_obs_saa =  y_obs_saa_all[i]
                    psi_obs_saa = psi_obs_saa_all[i]
                    vx_obs_saa =  vx_obs_saa_all[i]
                    vy_obs_saa =  vy_obs_saa_all[i]
                    prob_obs_saa = prob_obs_saa_all[i]
                    y_des_1_saa = y_des_1_saa_all[i]
                    y_des_2_saa = y_des_2_saa_all[i]

                    count_saa,_,_ = compute_stats(13*i+ 5*k + 17,y_des_1_saa[k],y_des_2_saa[k],
                                                prob_obs_saa[k],cx_all_saa[k],cy_all_saa[k],
                                            x_obs_saa[k],y_obs_saa[k],vx_obs_saa[k],vy_obs_saa[k])
                
                    count_saa_all[i][k] = count_saa

                    cx_all_cvar = cx_all_cvar_all[i]
                    cy_all_cvar = cy_all_cvar_all[i]
                    init_state_all_cvar = init_state_all_cvar_all[i]
                    x_obs_cvar =  x_obs_cvar_all[i]
                    y_obs_cvar =  y_obs_cvar_all[i]
                    psi_obs_cvar = psi_obs_cvar_all[i]
                    vx_obs_cvar =  vx_obs_cvar_all[i]
                    vy_obs_cvar =  vy_obs_cvar_all[i]
                    prob_obs_cvar = prob_obs_cvar_all[i]
                    y_des_1_cvar = y_des_1_cvar_all[i]
                    y_des_2_cvar = y_des_2_cvar_all[i]

                    count_cvar,_,_ = compute_stats(13*i+ 5*k + 17,y_des_1_cvar[k],y_des_2_cvar[k],
                                                prob_obs_cvar[k],cx_all_cvar[k],cy_all_cvar[k],
                                            x_obs_cvar[k],y_obs_cvar[k],vx_obs_cvar[k],vy_obs_cvar[k])
                
                    count_cvar_all[i][k] = count_cvar

            coll_rand = np.mean(count_rand_all,axis=0)
            coll_saa = np.mean(count_saa_all,axis=0)
            coll_cvar = np.mean(count_cvar_all,axis=0)
            coll_mmd = np.mean(count_mmd_all,axis=0)
            
            np.savez("./stats/dynamic/{}/stats_{}_samples_{}_obs".format(sc,num_reduced,num_obs), 
                    coll_mmd = coll_mmd,coll_saa=coll_saa,
                    coll_cvar = coll_cvar,coll_rand = coll_rand,
                    vel_mmd = vel_mmd,vel_saa=vel_saa,
                    vel_cvar = vel_cvar,vel_rand = vel_rand)

