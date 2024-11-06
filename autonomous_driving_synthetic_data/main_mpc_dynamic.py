import numpy as np
import sys
sys.path.insert(1, '/home/ims-robotics/MMD-OPT/autonomous_driving_synthetic_data/optimizer')
from optimizer import cem
import argparse
from obs_data_generate_dynamic import obs_data

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_reduced_set',type=int, nargs='+', required=True)
    parser.add_argument('--num_obs',type=int, nargs='+', required=True)
    parser.add_argument('--costs',type=str, nargs='+', required=True)
    parser.add_argument('--setting',type=str, nargs='+', required=True)
    parser.add_argument('--num_exps',type=int,required=True)
    args = parser.parse_args()
    
    list_num_reduced = args.num_reduced_set
    list_num_obs = args.num_obs 
    list_costs = args.costs
    list_settings = args.setting
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
    
    for setting in list_settings:
        if setting=="inlane_cut_in_low":
            prob_arr = np.array([0.05,0.1,0.15,0.2,0.25,0.3]) # cut in with low prob
        elif setting=="inlane_cut_in_high":
            prob_arr = np.array([0.95,0.9,0.85,0.8,0.75,0.7]) # cut in with high prob
        else:
            prob_arr = np.array([0.05,0.9,0.1,0.95,0.2,0.3,0.8,0.7,0.6,0.4]) # lane_change
        
        for pp,num_obs in enumerate(list_num_obs):
            for ii,num_reduced in enumerate(list_num_reduced):
                prob = cem.CEM(num_reduced,num_mother,num_obs,setting)
            #####################################################
                for cost in list_costs:
                    if cost=="mmd_opt":
                        func_cem = prob.compute_cem_mmd_opt
                        threshold = -prob.prob.ker_wt*prob.num_obs + 5
                        prob_obs_data = obs_data(num_mother)

                    elif cost=="mmd_random":
                        func_cem = prob.compute_cem_mmd_opt
                        threshold = -prob.prob.ker_wt*prob.num_obs + 5
                        prob_obs_data = obs_data(num_mother)
                        prob_obs_data_red = obs_data(num_reduced)
                    
                    elif cost=="saa":
                        func_cem = prob.compute_cem_saa
                        threshold = 1e-5
                        prob_obs_data = obs_data(num_reduced)

                    else:
                        func_cem = prob.compute_cem_cvar
                        threshold = 1e-5
                        prob_obs_data = obs_data(num_reduced)

                    for l in range(num_exps):   
                        cx_all,cy_all,init_state_all= np.zeros((0,prob.nvar)),\
                                        np.zeros((0,prob.nvar)),\
                                        np.zeros((0,6))

                        x_obs,y_obs =  np.zeros((prob.num_obs,prob.prob.num_reduced*prob.num_circles,prob.num)),\
                    np.zeros((prob.num_obs,prob.prob.num_reduced*prob.num_circles,prob.num))
                                
                        x_obs_all,y_obs_all, vx_obs_all,vy_obs_all,psi_obs_all = np.zeros((0,prob.num_obs)),np.zeros((0,prob.num_obs)),\
                        np.zeros((0,prob.num_obs)),np.zeros((0,prob.num_obs)),np.zeros((0,prob.num_obs))
                            
                        prob_all = np.zeros((0,num_obs))

                        y_des_1_all ,y_des_2_all = np.zeros((0,prob.num_obs)),np.zeros((0,prob.num_obs))

                        for k in range(num_configs): 
                            y_des_1_data,y_des_2_data = np.zeros(num_obs),np.zeros(num_obs)

                            x_obs_init,y_obs_init,vx_obs_init,vy_obs_init,\
                                psi_obs_init \
                                = prob_obs_data.compute_obs_data(num_obs,k)
                            
                            np.random.seed(k)
                            prob_des_1 = np.random.choice(prob_arr, (num_obs, ),replace=False)
                            prob_des_2 = 1. - prob_des_1

                            beta,sigma_best = np.zeros((num_obs,num_reduced)),np.zeros(num_obs)
                            
                            for tt in range(num_obs):
                                probabilities = np.hstack((prob_des_1[tt],prob_des_2[tt] ))

                                y_des_1 = -1.75 
                                y_des_2 = 1.75 
                                y_des= np.array([y_des_1,y_des_2]).reshape(-1)
                    
                                if cost=="mmd_opt":

                                    np.random.seed(23*l+43*k+ 11*tt + 113*pp + 97*ii +5)
                                    y_samples = np.random.choice(y_des, num_mother, p=list(probabilities))
                                    
                                    b_eq_x,b_eq_y = prob_obs_data.compute_boundary_vec(x_obs_init[tt],vx_obs_init[tt],
                                                                                    0.,y_obs_init[tt],vy_obs_init[tt],0.)
                                    
                                    _x_obs_traj,_y_obs_traj \
                                        = prob_obs_data.compute_obs_guess(b_eq_x,b_eq_y,y_samples,23*l+43*k+ 11*tt +5)
                    
                                    cx_obs_traj,cy_obs_traj = prob.prob.compute_coeff(_x_obs_traj,_y_obs_traj)
                                
                                    beta_best,res,_sigma_best,ker_red_best,\
                                    _x_obs_traj_red,_y_obs_traj_red \
                                        = prob.prob2.compute_cem(cx_obs_traj,cy_obs_traj,_x_obs_traj,_y_obs_traj)

                                    beta[tt] = beta_best.reshape(prob.prob.num_reduced)

                                    x_obs[tt] = _x_obs_traj_red
                                    y_obs[tt] = _y_obs_traj_red

                                    sigma_best[tt] = _sigma_best
                                    y_des_1_data[tt] = y_des_1
                                    y_des_2_data[tt] = y_des_2

                                elif cost=="mmd_random":

                                    np.random.seed(23*l+43*k+ 11*tt + 113*pp + 97*ii +5)
                                    y_samples = np.random.choice(y_des, num_mother, p=list(probabilities))
                                    
                                    b_eq_x,b_eq_y = prob_obs_data.compute_boundary_vec(x_obs_init[tt],vx_obs_init[tt],
                                                                                    0.,y_obs_init[tt],vy_obs_init[tt],0.)
                                    
                                    _x_obs_traj,_y_obs_traj \
                                        = prob_obs_data.compute_obs_guess(b_eq_x,b_eq_y,y_samples,23*l+43*k+ 11*tt +5)
                    
                                    cx_obs_traj,cy_obs_traj = prob.prob.compute_coeff(_x_obs_traj,_y_obs_traj)
                                    B = np.hstack((cx_obs_traj,cy_obs_traj ))

                                    y_samples = np.random.choice(y_des, num_reduced, p=list(probabilities))
                                    
                                    b_eq_x,b_eq_y = prob_obs_data_red.compute_boundary_vec(x_obs_init[tt],vx_obs_init[tt],
                                                                                    0.,y_obs_init[tt],vy_obs_init[tt],0.)
                                    
                                    _x_obs_traj_red,_y_obs_traj_red \
                                        = prob_obs_data_red.compute_obs_guess(b_eq_x,b_eq_y,y_samples,23*l+43*k+ 11*tt +5)
                    
                                    cx_obs_traj_red,cy_obs_traj_red = prob.prob.compute_coeff(_x_obs_traj_red,_y_obs_traj_red)
                                    A = np.hstack((cx_obs_traj_red,cy_obs_traj_red ))

                                    _sigma_best = 0.01

                                    ker_red,ker_mixed,ker_total = prob.prob.compute_kernel(A,B,_sigma_best)
                                    beta_opt,cost_exh = prob.prob2.compute_beta_reduced(ker_red,ker_mixed) # red_set

                                    beta[tt] = beta_opt.reshape(prob.prob.num_reduced)

                                    x_obs[tt] = _x_obs_traj_red
                                    y_obs[tt] = _y_obs_traj_red

                                    sigma_best[tt] = _sigma_best
                                    y_des_1_data[tt] = y_des_1
                                    y_des_2_data[tt] = y_des_2

                                else:
                                    np.random.seed(3*l+5*k+ 19*tt +23)
                                    y_samples = np.random.choice(y_des,num_reduced, p=list(probabilities))
                                    
                                    b_eq_x,b_eq_y = prob_obs_data.compute_boundary_vec(x_obs_init[tt],vx_obs_init[tt],0.,y_obs_init[tt],vy_obs_init[tt],0.)
                                    
                                    _x_obs_traj,_y_obs_traj = prob_obs_data.compute_obs_guess(b_eq_x,b_eq_y,y_samples,3*l+5*k+ 19*tt +23)
                                    
                                    _x_obs_traj_red = _x_obs_traj
                                    _y_obs_traj_red = _y_obs_traj

                                    x_obs[tt] = _x_obs_traj_red
                                    y_obs[tt] = _y_obs_traj_red
                                    y_des_1_data[tt] = y_des_1
                                    y_des_2_data[tt] = y_des_2

                            cx_best,cy_best,cost_obs = func_cem(k,init_state,mean_param,cov_param,
                                    x_obs,y_obs,v_des,beta,sigma_best)   
                
                            if cost_obs<= threshold:
                                cx_all = np.append(cx_all,cx_best.reshape(1,-1),axis=0)
                                cy_all = np.append(cy_all,cy_best.reshape(1,-1),axis=0)
                                init_state_all = np.append(init_state_all,init_state.reshape(1,-1),axis=0)
                                x_obs_all = np.append(x_obs_all,x_obs_init.reshape(1,-1),axis=0)
                                y_obs_all = np.append(y_obs_all,y_obs_init.reshape(1,-1),axis=0)
                                vx_obs_all = np.append(vx_obs_all,vx_obs_init.reshape(1,-1),axis=0)
                                vy_obs_all = np.append(vy_obs_all,vy_obs_init.reshape(1,-1),axis=0)
                                psi_obs_all = np.append(psi_obs_all,psi_obs_init.reshape(1,-1),axis=0)
                                prob_all = np.append(prob_all,prob_des_1.reshape(1,-1),axis=0)
                                y_des_1_all = np.append(y_des_1_all,y_des_1_data.reshape(1,-1),axis=0)
                                y_des_2_all = np.append(y_des_2_all,y_des_2_data.reshape(1,-1),axis=0)

                        np.savez("./data/dynamic/{}/{}_{}_samples_{}_obs_{}".format(setting,cost,num_reduced,num_obs,l), 
                                 cx = cx_all,cy = cy_all,
                    init_state = init_state_all,x_obs_all=x_obs_all,y_obs_all=y_obs_all,
                    vx_obs_all=vx_obs_all,vy_obs_all=vy_obs_all,psi_obs_all=psi_obs_all,prob = prob_all,
                    y_des_1 = y_des_1_all,y_des_2 = y_des_2_all)

                        print("cost {}, reduced_set {}, num_obs {}, experiment {} ".format(cost,num_reduced,num_obs,l))
                        print("--------------------------------")
            
if __name__ == '__main__':
    main()


