import numpy as np
from optimizer import cem

prob = cem.CEM(1,1,1,"static")

def compute_noisy_init_state_gaussian(x_init,y_init,psi_init,_num_batch,seed):
    np.random.seed(seed)

    mu = np.asarray([0, 0, 0.])
    sigma = np.asarray([1, 0.2, 0.1])  

    epsilon = np.random.multivariate_normal(np.zeros(3),np.eye(3), (_num_batch,) )
    epsilon_x = epsilon[:,0]
    epsilon_y = epsilon[:,1]
    epsilon_psi = epsilon[:,2]

    eps_x = epsilon_x*sigma[0] + mu[0]
    eps_y = epsilon_y*sigma[1] + mu[1]
    eps_psi = epsilon_psi*sigma[2] + mu[2]
    
    x_init = x_init + eps_x
    y_init = y_init + eps_y
    psi_init = psi_init + 0*eps_psi

    return x_init,y_init,psi_init

def compute_noisy_init_state_bimodal(x_init,y_init,psi_init,_num_batch,seed):
    np.random.seed(seed)

    modes = np.array([1,2])
    modes_probs = np.array([0.8,0.2])
    mu = np.asarray([[-1, -0.3, 0.],[1, 0.3, 0.01]])
    sigma = np.asarray([[0.5, 0.1, 0.1],[0.1, 0.02, 0.05]])  

    epsilon = np.random.multivariate_normal(np.zeros(3),np.eye(3), (_num_batch,) )
    epsilon_x = epsilon[:,0]
    epsilon_y = epsilon[:,1]
    epsilon_psi = epsilon[:,2]

    eps_x_1 = epsilon_x*sigma[0][0] + mu[0][0]
    eps_x_2 = epsilon_x*sigma[1][0] + mu[1][0]

    eps_y_1 = epsilon_y*sigma[0][1] + mu[0][1]
    eps_y_2 = epsilon_y*sigma[1][1] + mu[1][1]

    eps_psi_1 = epsilon_psi*sigma[0][2] + mu[0][2]
    eps_psi_2 = epsilon_psi*sigma[1][2] + mu[1][2]

    weight_samples = np.random.choice(modes, (_num_batch,),p=modes_probs)

    idx_1 = np.where(weight_samples==1)
    idx_2 = np.where(weight_samples==2)

    eps_x = np.hstack((eps_x_1[idx_1],eps_x_2[idx_2]))
    eps_y = np.hstack((eps_y_1[idx_1],eps_y_2[idx_2]))
    eps_psi = np.hstack((eps_psi_1[idx_1],eps_psi_2[idx_2]))
    
    x_init = x_init + eps_x
    y_init = y_init + eps_y
    psi_init = psi_init + 0*eps_psi

    return x_init,y_init,psi_init

def compute_noisy_init_state_trimodal(x_init,y_init,psi_init,_num_batch,seed):
    np.random.seed(seed)

    modes = np.array([1,2,3])
    modes_probs = np.array([0.4,0.2,0.4])
    mu = np.asarray([[-1, -0.3, 0.],[1, 0.3, 0.01],[-1, 0.2, -0.1]])
    sigma = np.asarray([[0.5, 0.1, 0.1],[0.1, 0.02, 0.05],[0.1, 0.02, 0.01]])  

    epsilon = np.random.multivariate_normal(np.zeros(3),np.eye(3), (_num_batch,) )
    epsilon_x = epsilon[:,0]
    epsilon_y = epsilon[:,1]
    epsilon_psi = epsilon[:,2]

    eps_x_1 = epsilon_x*sigma[0][0] + mu[0][0]
    eps_x_2 = epsilon_x*sigma[1][0] + mu[1][0]
    eps_x_3 = epsilon_x*sigma[2][0] + mu[2][0]

    eps_y_1 = epsilon_y*sigma[0][1] + mu[0][1]
    eps_y_2 = epsilon_y*sigma[1][1] + mu[1][1]
    eps_y_3 = epsilon_y*sigma[2][1] + mu[2][1]

    eps_psi_1 = epsilon_psi*sigma[0][2] + mu[0][2]
    eps_psi_2 = epsilon_psi*sigma[1][2] + mu[1][2]
    eps_psi_3 = epsilon_psi*sigma[2][2] + mu[2][2]

    weight_samples = np.random.choice(modes, (_num_batch,),p=modes_probs)

    idx_1 = np.where(weight_samples==1)
    idx_2 = np.where(weight_samples==2)
    idx_3 = np.where(weight_samples==3)

    eps_x = np.hstack((eps_x_1[idx_1],eps_x_2[idx_2],eps_x_3[idx_3]))
    eps_y = np.hstack((eps_y_1[idx_1],eps_y_2[idx_2],eps_y_3[idx_3]))
    eps_psi = np.hstack((eps_psi_1[idx_1],eps_psi_2[idx_2],eps_psi_3[idx_3]))
    
    x_init = x_init + eps_x
    y_init = y_init + eps_y
    psi_init = psi_init + 0*eps_psi

    return x_init,y_init,psi_init

def compute_obs_trajectories(x_obs,y_obs,psi_obs):
    _num_batch = x_obs.shape[0]

    x_obs_init = x_obs
    y_obs_init = y_obs

    vx_obs_init  = np.zeros(_num_batch)
    vy_obs_init  = np.zeros(_num_batch)

    x_obs_traj = (x_obs_init + vx_obs_init * prob.tot_time[:, np.newaxis]).T # num_obs x num
    y_obs_traj = (y_obs_init + vy_obs_init * prob.tot_time[:, np.newaxis]).T
    
    psi_obs_traj = np.tile(psi_obs, (prob.num,1)).T # num_obs x num
    
    return x_obs_traj,y_obs_traj,psi_obs_traj

def compute_obs_data(num_obs,seed):
    np.random.seed(seed)

    x_obs_init = np.random.choice(np.linspace(30,100,10), (num_obs, ),replace=False)
    y_obs_init = np.random.choice(np.array([-1.75,1.75]),(num_obs,))
    psi_obs_init = np.zeros(num_obs)

    return x_obs_init,y_obs_init,psi_obs_init

