import jax.numpy as jnp
from functools import partial
from jax import jit, random,vmap
import jax

class Helper():
    def __init__(self,v_min,v_max, P,Pdot,Pddot,rho_v,rho_offset,
                 weight_smoothness_x,weight_smoothness_y,k_p_v,k_d_v,k_p,k_d,
                 P_jax_1,P_jax_2,P_jax_3,P_jax_4,
                 Pdot_jax_1,Pdot_jax_2,Pdot_jax_3,Pdot_jax_4,
                 Pddot_jax_1,Pddot_jax_2,Pddot_jax_3,Pddot_jax_4,
                 num_params,num_batch,ellite_num,num_partial,alpha_mean,alpha_cov,
                 lamda,
                 tot_time,wheel_base,num_reduced,
                 num_prime,num_circles,num_obs,A_eq_x,A_eq_y,y_des_1,y_des_2,ellite_num_cost,num,
                 steer_max,nvar,dt,num_mother,prob,prob2):
        
        self.prob = prob
        self.prob2 = prob2
        self.num_mother = num_mother

        self.t = dt
        self.nvar = nvar
        self.steer_max = steer_max
        self.num_prime = num_prime
        self.num_obs = num_obs
        self.num_circles = num_circles
        self.num = num
        self.ellite_num_cost = ellite_num_cost
        self.y_des_1, self.y_des_2 = y_des_1,y_des_2

        self.tot_time = tot_time
        self.wheel_base = wheel_base
        self.num_reduced = num_reduced

        self.v_max = v_max
        self.v_min = v_min
    
        self.P_jax, self.Pdot_jax, self.Pddot_jax = P,Pdot,Pddot

        self.rho_v = rho_v 
        self.rho_offset = rho_offset
        
        self.weight_smoothness_x = weight_smoothness_x
        self.weight_smoothness_y = weight_smoothness_y

        self.k_p_v = k_p_v
        self.k_d_v = k_d_v

        self.k_p = k_p
        self.k_d = k_d
        self.A_eq_x = A_eq_x
        self.A_eq_y = A_eq_y

        self.P_jax_1 = P_jax_1
        self.P_jax_2 = P_jax_2
        self.P_jax_3 = P_jax_3
        self.P_jax_4 = P_jax_4

        self.Pdot_jax_1 = Pdot_jax_1
        self.Pdot_jax_2 = Pdot_jax_2
        self.Pdot_jax_3 = Pdot_jax_3
        self.Pdot_jax_4 = Pdot_jax_4
            
        self.Pddot_jax_1 = Pddot_jax_1
        self.Pddot_jax_2 = Pddot_jax_2
        self.Pddot_jax_3 = Pddot_jax_3
        self.Pddot_jax_4 = Pddot_jax_4

        self.num_partial = num_partial
        ###########################################3
        key = random.PRNGKey(0)

        self.key = key

        self.alpha_mean = alpha_mean
        self.alpha_cov = alpha_cov

        self.lamda = lamda
        self.vec_product = jit(jax.vmap(self.comp_prod, 0, out_axes=(0)))

        self.num_params = num_params 
        self.num_batch = num_batch
        self.ellite_num = ellite_num

        self.compute_controls_vmap = jit(vmap(self.compute_controls,in_axes=(0,0)))

    @partial(jit, static_argnums=(0,))	
    def sampling_param(self,mean_param, cov_param):

        key, subkey = random.split(self.key)
        param_samples = jax.random.multivariate_normal(key, mean_param, cov_param, (self.num_batch,))
        
        v_des_1 = param_samples[:, 0]
        v_des_2 = param_samples[:, 1]
        v_des_3 = param_samples[:, 2]
        v_des_4 = param_samples[:, 3]
            
        y_des_1 = param_samples[:, 4]
        y_des_2 = param_samples[:, 5]
        y_des_3 = param_samples[:, 6]
        y_des_4 = param_samples[:, 7]

        v_des_1 = jnp.clip(v_des_1, self.v_min*jnp.ones(self.num_batch),
                            self.v_max*jnp.ones(self.num_batch)   )
        v_des_2 = jnp.clip(v_des_2, self.v_min*jnp.ones(self.num_batch), 
                           self.v_max*jnp.ones(self.num_batch)   )
        v_des_3 = jnp.clip(v_des_3, self.v_min*jnp.ones(self.num_batch), 
                           self.v_max*jnp.ones(self.num_batch)   )
        v_des_4 = jnp.clip(v_des_4, self.v_min*jnp.ones(self.num_batch),
                            self.v_max*jnp.ones(self.num_batch)   )

        neural_output_batch = jnp.vstack(( v_des_1, v_des_2, v_des_3, v_des_4, y_des_1, y_des_2, y_des_3, y_des_4,
                                          )).T
        
        return neural_output_batch
	
    @partial(jit, static_argnums=(0,))	
    def compute_boundary_vec(self, x_init, vx_init, ax_init, y_init, vy_init, ay_init):

        x_init_vec = x_init*jnp.ones((self.num_batch, 1))
        y_init_vec = y_init*jnp.ones((self.num_batch, 1)) 

        vx_init_vec = vx_init*jnp.ones((self.num_batch, 1))
        vy_init_vec = vy_init*jnp.ones((self.num_batch, 1))

        ax_init_vec = ax_init*jnp.ones((self.num_batch, 1))
        ay_init_vec = ay_init*jnp.ones((self.num_batch, 1))

        b_eq_x = jnp.hstack(( x_init_vec, vx_init_vec, ax_init_vec ))
        b_eq_y = jnp.hstack(( y_init_vec, vy_init_vec, ay_init_vec, jnp.zeros((self.num_batch, 1  ))   ))

        return b_eq_x, b_eq_y

    @partial(jit, static_argnums=(0,))	
    def compute_x_guess(self, b_eq_x, b_eq_y, neural_output_batch):

        v_des_1 = neural_output_batch[:, 0]
        v_des_2 = neural_output_batch[:, 1]
        v_des_3 = neural_output_batch[:, 2]
        v_des_4 = neural_output_batch[:, 3]

        y_des_1 = neural_output_batch[:, 4]
        y_des_2 = neural_output_batch[:, 5]
        y_des_3 = neural_output_batch[:, 6]
        y_des_4 = neural_output_batch[:, 7]
        
        #############################
        A_vd_1 = self.Pddot_jax_1-self.k_p_v*self.Pdot_jax_1 #- self.k_d_v*self.Pddot_jax_1
        b_vd_1 = -self.k_p_v*jnp.ones((self.num_batch, self.num_partial ))*(v_des_1)[:, jnp.newaxis]

        A_vd_2 = self.Pddot_jax_2-self.k_p_v*self.Pdot_jax_2 #- self.k_d_v*self.Pddot_jax_2
        b_vd_2 = -self.k_p_v*jnp.ones((self.num_batch, self.num_partial ))*(v_des_2)[:, jnp.newaxis]

        A_vd_3 = self.Pddot_jax_3-self.k_p_v*self.Pdot_jax_3 #- self.k_d_v*self.Pddot_jax_3
        b_vd_3 = -self.k_p_v*jnp.ones((self.num_batch, self.num_partial ))*(v_des_3)[:, jnp.newaxis]

        A_vd_4 = self.Pddot_jax_4-self.k_p_v*self.Pdot_jax_4 #- self.k_d_v*self.Pddot_jax_4
        b_vd_4 = -self.k_p_v*jnp.ones((self.num_batch, self.num_partial ))*(v_des_4)[:, jnp.newaxis]

        A_pd_1 = self.Pddot_jax_1-self.k_p*self.P_jax_1# -self.k_d*self.Pdot_jax_1
        b_pd_1 = -self.k_p*jnp.ones((self.num_batch, self.num_partial ))*(y_des_1)[:, jnp.newaxis]
        
        A_pd_2 = self.Pddot_jax_2-self.k_p*self.P_jax_2 #-self.k_d*self.Pdot_jax_2
        b_pd_2 = -self.k_p*jnp.ones((self.num_batch, self.num_partial ))*(y_des_2)[:, jnp.newaxis]
            
        A_pd_3 = self.Pddot_jax_3-self.k_p*self.P_jax_3 #-self.k_d*self.Pdot_jax_3
        b_pd_3 = -self.k_p*jnp.ones((self.num_batch, self.num_partial ))*(y_des_3)[:, jnp.newaxis]
        
        A_pd_4 = self.Pddot_jax_4-self.k_p*self.P_jax_4 #-self.k_d*self.Pdot_jax_4
        b_pd_4 = -self.k_p*jnp.ones((self.num_batch, self.num_partial ))*(y_des_4)[:, jnp.newaxis]
        
        cost_smoothness_x = self.weight_smoothness_x*jnp.dot(self.Pddot_jax.T, self.Pddot_jax)
        cost_smoothness_y = self.weight_smoothness_y*jnp.dot(self.Pddot_jax.T, self.Pddot_jax)
        
        cost_temp_x = self.rho_v*jnp.dot(A_vd_1.T, A_vd_1)+self.rho_v*jnp.dot(A_vd_2.T, A_vd_2)+self.rho_v*jnp.dot(A_vd_3.T, A_vd_3)+self.rho_v*jnp.dot(A_vd_4.T, A_vd_4)
        cost_temp_y = self.rho_offset*jnp.dot(A_pd_1.T, A_pd_1)+self.rho_offset*jnp.dot(A_pd_2.T, A_pd_2)+self.rho_offset*jnp.dot(A_pd_3.T, A_pd_3)+self.rho_offset*jnp.dot(A_pd_4.T, A_pd_4)
        
        cost_x = cost_smoothness_x + cost_temp_x
        cost_y = cost_smoothness_y + cost_temp_y

        cost_mat_x = jnp.vstack((  jnp.hstack(( cost_x, self.A_eq_x.T )), jnp.hstack(( self.A_eq_x, jnp.zeros(( jnp.shape(self.A_eq_x)[0], jnp.shape(self.A_eq_x)[0] )) )) ))
        cost_mat_y = jnp.vstack((  jnp.hstack(( cost_y, self.A_eq_y.T )), jnp.hstack(( self.A_eq_y, jnp.zeros(( jnp.shape(self.A_eq_y)[0], jnp.shape(self.A_eq_y)[0] )) )) ))
        
        lincost_x = -self.rho_v*jnp.dot(A_vd_1.T, b_vd_1.T).T-self.rho_v*jnp.dot(A_vd_2.T, b_vd_2.T).T-self.rho_v*jnp.dot(A_vd_3.T, b_vd_3.T).T-self.rho_v*jnp.dot(A_vd_4.T, b_vd_4.T).T
        lincost_y = -self.rho_offset*jnp.dot(A_pd_1.T, b_pd_1.T).T-self.rho_offset*jnp.dot(A_pd_2.T, b_pd_2.T).T-self.rho_offset*jnp.dot(A_pd_3.T, b_pd_3.T).T-self.rho_offset*jnp.dot(A_pd_4.T, b_pd_4.T).T 
    
        sol_x = jnp.linalg.solve(cost_mat_x, jnp.hstack(( -lincost_x, b_eq_x )).T).T
        sol_y = jnp.linalg.solve(cost_mat_y, jnp.hstack(( -lincost_y, b_eq_y )).T).T

        #######################3

        primal_sol_x = sol_x[:,0:self.nvar]
        primal_sol_y = sol_y[:,0:self.nvar]

        return primal_sol_x, primal_sol_y

    @partial(jit, static_argnums=(0, ))
    def compute_cost(self,cost_obs,x_ellite_cost,y_ellite_cost, res_ellite_cost, \
        xdot_ellite_cost, ydot_ellite_cost, 
        xddot_ellite_cost, yddot_ellite_cost, \
             v_des, steering_ellite_cost):
        
        cost_centerline_1 = jnp.linalg.norm(y_ellite_cost - self.y_des_1,axis=1)
        cost_centerline_2 = jnp.linalg.norm(y_ellite_cost - self.y_des_2,axis=1)
        cost_des_lane = cost_centerline_1#*cost_centerline_2

        cost_steering = jnp.linalg.norm(steering_ellite_cost, axis = 1)
        steering_vel = jnp.diff(steering_ellite_cost, axis = 1)
        cost_steering_vel = jnp.linalg.norm(steering_vel, axis = 1)
        steering_acc = jnp.diff(steering_vel, axis = 1)
        
        v = jnp.sqrt(xdot_ellite_cost**2+ydot_ellite_cost**2)
    
        cost_steering_acc = jnp.linalg.norm(steering_acc, axis = 1)
        cost_steering_penalty = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.ellite_num_cost, self.num  )), jnp.abs(steering_ellite_cost)-self.steer_max ), axis = 1)
        cost_steering_vel_penalty = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.ellite_num_cost, self.num-1  )), jnp.abs(steering_vel)-0.05 ), axis = 1)
        
        cost_batch = res_ellite_cost \
            + 0.01*jnp.linalg.norm(v-v_des, axis = 1)\
                + cost_obs \
                + 0.1*(cost_steering + cost_steering_vel) \
                  + 0.1*(cost_steering_penalty+cost_steering_vel_penalty)\
                 + 0.1*cost_steering_acc\
                 + 0.02*(jnp.linalg.norm(yddot_ellite_cost, axis = 1)) \
                 + 0.02*(jnp.linalg.norm(xddot_ellite_cost, axis = 1)) \
                  + 0.01*cost_des_lane

        return cost_batch
    
    @partial(jit, static_argnums=(0, ))
    def compute_ellite_samples(self, cost_batch, neural_output_projection):

        idx_ellite = jnp.argsort(cost_batch)

        neural_output_ellite = neural_output_projection[idx_ellite[0:self.ellite_num]]

        return neural_output_ellite, idx_ellite
    
    @partial(jit, static_argnums=(0,))
    def comp_prod(self, diffs, d ):
        term_1 = jnp.expand_dims(diffs, axis = 1)
        term_2 = jnp.expand_dims(diffs, axis = 0)
        prods = d * jnp.outer(term_1,term_2)
        return prods	
  
    @partial(jit, static_argnums=(0, ))
    def compute_shifted_samples(self, key, neural_output_ellite, cost_batch, idx_ellite, mean_param_prev, cov_param_prev):
        cost_batch_temp = cost_batch[idx_ellite[0:self.ellite_num]]
        w = cost_batch_temp
        w_min = jnp.min(cost_batch_temp)
        w = jnp.exp(-(1/self.lamda) * (w - w_min ) )
        sum_w = jnp.sum(w, axis = 0)
        mean_param = (1-self.alpha_mean)*mean_param_prev + self.alpha_mean*(jnp.sum( (neural_output_ellite * w[:,jnp.newaxis]) , axis= 0)/ sum_w)
        diffs = (neural_output_ellite - mean_param)
        prod_result = self.vec_product(diffs, w)
        cov_param = (1-self.alpha_cov)*cov_param_prev + self.alpha_cov*(jnp.sum( prod_result , axis = 0)/jnp.sum(w, axis = 0)) + 0.01*jnp.identity(self.num_params)

        param_samples = jax.random.multivariate_normal(key, mean_param, cov_param, (self.num_batch-self.ellite_num, ))

        v_des_1 = param_samples[:, 0]
        v_des_2 = param_samples[:, 1]
        v_des_3 = param_samples[:, 2]
        v_des_4 = param_samples[:, 3]
        
        y_des_1 = param_samples[:, 4]
        y_des_2 = param_samples[:, 5]
        y_des_3 = param_samples[:, 6]
        y_des_4 = param_samples[:, 7]
                
        v_des_1 = jnp.clip(v_des_1, self.v_min*jnp.ones(self.num_batch-self.ellite_num), self.v_max*jnp.ones(self.num_batch-self.ellite_num)   )
        v_des_2 = jnp.clip(v_des_2, self.v_min*jnp.ones(self.num_batch-self.ellite_num), self.v_max*jnp.ones(self.num_batch-self.ellite_num)   )
        v_des_3 = jnp.clip(v_des_3, self.v_min*jnp.ones(self.num_batch-self.ellite_num), self.v_max*jnp.ones(self.num_batch-self.ellite_num)   )
        v_des_4 = jnp.clip(v_des_4, self.v_min*jnp.ones(self.num_batch-self.ellite_num), self.v_max*jnp.ones(self.num_batch-self.ellite_num)   )

        neural_output_shift = jnp.vstack(( v_des_1, v_des_2, v_des_3, v_des_4, y_des_1, y_des_2, y_des_3, y_des_4,
                                           )).T

        neural_output_batch = jnp.vstack(( neural_output_ellite, neural_output_shift  ))

        return mean_param, cov_param, neural_output_batch, cost_batch_temp
   
    @partial(jit, static_argnums=(0,))	
    def compute_obs_trajectories(self,x_obs,y_obs,vx_obs,vy_obs,psi_obs):

        x_obs_traj = (x_obs + vx_obs * self.tot_time[:, jnp.newaxis]).T # num_obs x num
        y_obs_traj = (y_obs + vy_obs * self.tot_time[:, jnp.newaxis]).T
        
        psi_obs_traj = jnp.tile(psi_obs, (self.num,1)).T # num_obs x num
        
        x_obs_circles = x_obs_traj
        y_obs_circles = y_obs_traj
        psi_obs_circles = psi_obs_traj

        return x_obs_circles,y_obs_circles,psi_obs_circles

    @partial(jit, static_argnums=(0, ))	
    def compute_controls(self, xdot,ydot,xddot,yddot):
        v = jnp.sqrt(xdot**2+ydot**2)
        v = jnp.hstack((v,v[:,-1].reshape(-1,1)))
        
        acc = jnp.diff(v,axis=1)/self.t
        acc = jnp.hstack((acc,acc[:,-1].reshape(-1,1)))

        curvature_best = (yddot*xdot-ydot*xddot)/((xdot**2+ydot**2)**(1.5)) 
        steer = jnp.arctan(curvature_best*self.wheel_base  )

        return acc,steer
    