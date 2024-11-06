import numpy as np
import sys
sys.path.insert(1, '/home/ims-robotics/MMD-OPT/autonomous_driving_synthetic_data/optimizer')
from optimizer import cem
import argparse
from obs_data_generate_static import compute_obs_data,compute_obs_trajectories

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_reduced_set',type=int, nargs='+', required=True)
    parser.add_argument('--num_obs',type=int, nargs='+', required=True)
    parser.add_argument('--costs',type=str, nargs='+', required=True)
    parser.add_argument('--noises',type=str, nargs='+', required=True)
    parser.add_argument("--num_exps",  type=int, required=True)

    args = parser.parse_args()
    
    list_noises = args.noises
    list_num_reduced = args.num_reduced_set
    list_num_obs = args.num_obs 
    list_costs = args.costs
    num_exps = args.num_exps
    
    x_init = 0.0
    vx_init = 3
    ax_init = 0.0

    y_init = -1.75
    vy_init = 0.
    ay_init = 0.0
    
    init_state = np.hstack(( x_init, y_init, vx_init, vy_init, ax_init, ay_init   ))

    v_des = 10.
    
    mean_vx_1 = v_des
    mean_vx_2 = v_des
    mean_vx_3 = v_des
    mean_vx_4 = v_des
    
    mean_y = 0.
    mean_y_des_1 = mean_y
    mean_y_des_2 = mean_y
    mean_y_des_3 = mean_y
    mean_y_des_4 = mean_y
   
    cov_vel = 20.
    cov_y = 100.
    mean_param = np.hstack(( mean_vx_1, mean_vx_2, mean_vx_3, mean_vx_4, mean_y_des_1, mean_y_des_2, mean_y_des_3, mean_y_des_4))

    diag_param = np.hstack(( cov_vel,cov_vel,cov_vel,cov_vel, cov_y,cov_y,cov_y,cov_y ))
    cov_param = np.asarray(np.diag(diag_param)) 
    
    num_mother = 100
    num_configs = 200

    for noise in list_noises:
        if noise == "gaussian":
            from obs_data_generate_static import compute_noisy_init_state_gaussian as compute_noisy_init_state
        elif noise == "bimodal":
            from obs_data_generate_static import compute_noisy_init_state_bimodal as compute_noisy_init_state
        else:
            from obs_data_generate_static import compute_noisy_init_state_trimodal as compute_noisy_init_state

        for pp,num_obs in enumerate(list_num_obs):
            for ii,num_reduced in enumerate(list_num_reduced):
                prob = cem.CEM(num_reduced,num_mother,num_obs,"static")
            #####################################################
                for cost in list_costs:
                    if cost=="mmd_opt" or cost=="mmd_random":
                        func_cem = prob.compute_cem_mmd_opt
                        threshold = -prob.prob.ker_wt*prob.num_obs + 5
                        
                    elif cost=="saa":
                        func_cem = prob.compute_cem_saa
                        threshold = 1e-5
                        
                    else:
                        func_cem = prob.compute_cem_cvar
                        threshold = 1e-5

                    for l in range(num_exps):   
                        cx_all,cy_all= np.zeros((0,prob.nvar)),\
                                    np.zeros((0,prob.nvar))
                        
                        x_obs,y_obs =  np.zeros((prob.num_obs,prob.prob.num_reduced*prob.num_circles,prob.num)),\
                    np.zeros((prob.num_obs,prob.prob.num_reduced*prob.num_circles,prob.num))
                        
                        sigma_best = np.zeros(prob.num_obs)
                          
                        x_obs_all,y_obs_all, psi_obs_all = np.zeros((0,prob.num_obs)),np.zeros((0,prob.num_obs)),\
                        np.zeros((0,prob.num_obs))

                        for k in range(num_configs): 

                            x_obs_init,y_obs_init,psi_obs_init \
                                = compute_obs_data(num_obs,k)
                            
                            beta,sigma_best = np.zeros((num_obs,num_reduced)),np.zeros(num_obs)
                            
                            for tt in range(num_obs):

                                if cost=="mmd_opt":

                                    _x_obs_init,_y_obs_init,_psi_obs_init \
                                    = compute_noisy_init_state(x_obs_init[tt],y_obs_init[tt],psi_obs_init[tt],num_mother,
                                                               19*pp+23*ii+ 5*l+11*k+17*tt+13)
                    
                                    _x_obs_traj,_y_obs_traj,_ = compute_obs_trajectories(_x_obs_init,_y_obs_init,_psi_obs_init)

                                    cx_obs_traj,cy_obs_traj = prob.prob.compute_coeff(_x_obs_traj,_y_obs_traj)

                                    beta_best,res,_sigma_best,ker_red_best,\
                                    _x_obs_traj_red,_y_obs_traj_red \
                                        = prob.prob2.compute_cem(cx_obs_traj,cy_obs_traj,_x_obs_traj,_y_obs_traj)

                                    beta[tt] = beta_best.reshape(prob.prob.num_reduced)

                                    x_obs[tt] = _x_obs_traj_red
                                    y_obs[tt] = _y_obs_traj_red

                                    sigma_best[tt] = _sigma_best

                                elif cost=="mmd_random":

                                    _x_obs_init,_y_obs_init,_psi_obs_init \
                                    = compute_noisy_init_state(x_obs_init[tt],y_obs_init[tt],psi_obs_init[tt],num_mother,
                                                               19*pp+23*ii+ 5*l+11*k+17*tt+13)
                    
                                    _x_obs_traj,_y_obs_traj,_ = compute_obs_trajectories(_x_obs_init,_y_obs_init,_psi_obs_init)

                                    cx_obs_traj,cy_obs_traj = prob.prob.compute_coeff(_x_obs_traj,_y_obs_traj)

                                    B = np.hstack((cx_obs_traj,cy_obs_traj ))

                                    _x_obs_init,_y_obs_init,_psi_obs_init \
                                    = compute_noisy_init_state(x_obs_init[tt],y_obs_init[tt],psi_obs_init[tt],num_reduced,
                                                               19*pp+23*ii+ 5*l+11*k+17*tt+13)
                    
                                    _x_obs_traj_red,_y_obs_traj_red,_ = compute_obs_trajectories(_x_obs_init,_y_obs_init,_psi_obs_init)

                                    cx_obs_traj_red,cy_obs_traj_red = prob.prob.compute_coeff(_x_obs_traj_red,_y_obs_traj_red)

                                    A = np.hstack((cx_obs_traj_red,cy_obs_traj_red ))

                                    _sigma_best = 0.01

                                    ker_red,ker_mixed,ker_total = prob.prob.compute_kernel(A,B,_sigma_best)
                                    beta_opt,cost_exh = prob.prob2.compute_beta_reduced(ker_red,ker_mixed) # red_set
                                    
                                    beta = beta.at[tt].set(beta_opt.reshape(prob.num_reduced))

                                    x_obs[tt] = _x_obs_traj_red
                                    y_obs[tt] = _y_obs_traj_red
                                    
                                    sigma_best[tt] = _sigma_best

                                else:
                                    _x_obs_init,_y_obs_init,_psi_obs_init \
                                    = compute_noisy_init_state(x_obs_init[tt],y_obs_init[tt],psi_obs_init[tt],num_reduced,
                                                               19*pp+23*ii+ 5*l+11*k+17*tt+13)
                    
                                    _x_obs_traj,_y_obs_traj,_ = compute_obs_trajectories(_x_obs_init,_y_obs_init,_psi_obs_init)

                                    _x_obs_traj_red = _x_obs_traj
                                    _y_obs_traj_red = _y_obs_traj
                    
                                    x_obs[tt] = _x_obs_traj_red
                                    y_obs[tt] = _y_obs_traj_red
                                    
                            cx_best,cy_best,cost_obs = func_cem(k,init_state,mean_param,cov_param,
                                    x_obs,y_obs,v_des,beta,sigma_best)

                            if cost_obs<= threshold:
                                cx_all = np.append(cx_all,cx_best.reshape(1,-1),axis=0)
                                cy_all = np.append(cy_all,cy_best.reshape(1,-1),axis=0)
                                x_obs_all = np.append(x_obs_all,x_obs_init.reshape(1,-1),axis=0)
                                y_obs_all = np.append(y_obs_all,y_obs_init.reshape(1,-1),axis=0)
                                psi_obs_all = np.append(psi_obs_all,psi_obs_init.reshape(1,-1),axis=0)
                            
                        np.savez("./data/static/{}/{}_{}_samples_{}_obs_{}".format(noise,
                            cost,num_reduced,num_obs,l),
                                cx = cx_all,cy = cy_all,
                            x_obs=x_obs_all,y_obs=y_obs_all)

                        print("cost {}, reduced_set {}, num_obs {}, noise {}, experiment {} ".format(cost,num_reduced,num_obs,noise,l))
                        print("--------------------------------")
                
if __name__ == '__main__':
    main()


