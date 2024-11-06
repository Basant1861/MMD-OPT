import jax.numpy as jnp
from functools import partial
from jax import jit,vmap
import jax

class Costs():
    def __init__(self,prob,num_reduced,num_obs,num_prime,a_obs,b_obs,
                 y_lb,y_ub,alpha_quant,alpha_quant_lane,y_des_1,y_des_2,ellite_num_projection):
        
        self.ellite_num_projection = ellite_num_projection
        
        self.y_des_1,self.y_des_2 = y_des_1, y_des_2
        self.alpha_quant = alpha_quant
        self.alpha_quant_lane = alpha_quant_lane

        self.prob = prob
        self.y_lb,self.y_ub = y_lb,y_ub

        self.num_prime = num_prime
        self.num_reduced = num_reduced
        self.num_obs = num_obs
        self.a_obs,self.b_obs = a_obs,b_obs
        
        self.compute_f_bar_vmap = jit(vmap(self.compute_f_bar,in_axes=(0,0,None,None)))

        self.compute_mmd_obs_vmap = jit(vmap(self.compute_mmd_obs,
                        in_axes=(None,0,0,None,None,None)))
        
        self.compute_cvar_obs_vmap = jit(vmap(self.compute_cvar_obs,
                        in_axes=(0,0,None,None)))
        
        self.compute_saa_obs_vmap = jit(vmap(self.compute_saa_obs,
                        in_axes=(0,0,None,None)))

    @partial(jit, static_argnums=(0,))
    def compute_f_bar(self,x,y,x_obs,y_obs): 

        wc_alpha = (x-x_obs)
        ws_alpha = (y-y_obs)

        cost = -(wc_alpha**2)/(self.a_obs**2) - (ws_alpha**2)/(self.b_obs**2) + jnp.ones((self.num_obs,self.num_reduced,self.num_prime))
        
        cost_bar = jnp.maximum(jnp.zeros((self.num_obs,self.num_reduced,self.num_prime)), cost)

        return cost_bar # num_obs x num_reduced x num_prime
    
    @partial(jit, static_argnums=(0, ))	
    def compute_mmd_obs(self,beta,x_roll,y_roll,x_obs,y_obs,sigma):

        mmd_total = jnp.zeros((self.num_obs,self.ellite_num_projection))
       
        cost_mmd = self.compute_f_bar_vmap(x_roll[:,0:self.num_prime],y_roll[:,0:self.num_prime],
                                                x_obs,y_obs) # ellite_num_projection x num_obs x num_reduced x num
        
        for k in range(self.num_obs):
            _cost_mmd = cost_mmd[:,k,:,:] # (ellite_num_projection x num_reduced x num)
            _cost_mmd = jnp.max(_cost_mmd,axis=2) # (ellite_num_projection x num_reduced)

            mmd_total = mmd_total.at[k].set(self.prob.compute_mmd_vmap(beta[k],_cost_mmd,sigma[k]).reshape(-1))

        return  jnp.sum(mmd_total,axis=0)
       
    @partial(jit, static_argnums=(0, ))	
    def compute_cvar_obs(self,x_roll,y_roll,
                    x_obs,y_obs):
        
        costbar = self.compute_f_bar_vmap(x_roll[:,0:self.num_prime],y_roll[:,0:self.num_prime],
                                                x_obs,y_obs) # ellite_num_projection x num_obs x num_reduced x num
        
        costbar = jnp.max(jnp.max(costbar,axis=3),axis=1) # ellite_num_projection x num_reduced 
       
        var_alpha = jnp.quantile(costbar,self.alpha_quant, axis=1) # ellite_num_projection
        cvar_alpha = vmap(lambda i: jnp.where(costbar[i]>=var_alpha[i],costbar[i],
                                              jnp.full((costbar[i].shape[0],),jnp.nan)))\
                                                (jnp.arange(self.ellite_num_projection)) # ellite_num_projection x num_reduced 
        
        num_cvar = jnp.count_nonzero(~jnp.isnan(cvar_alpha),axis=1)
        cvar_alpha = jnp.nan_to_num(cvar_alpha)

        cvar_alpha = vmap(lambda i: jax.lax.cond( num_cvar[i]>0, lambda _: cvar_alpha[i].sum()/num_cvar[i], lambda _ : 0., 0. ))\
                                                                                                    (jnp.arange(self.ellite_num_projection))
      
        return cvar_alpha
    
    @partial(jit, static_argnums=(0, ))	
    def compute_saa_obs(self,x_roll,y_roll,
                    x_obs,y_obs):
        
        costbar = self.compute_f_bar_vmap(x_roll[:,0:self.num_prime],y_roll[:,0:self.num_prime],
                                                x_obs,y_obs) # ellite_num_projection x num_obs x num_reduced x num
        
        costbar = jnp.max(jnp.max(costbar,axis=3),axis=1) # ellite_num_projection x num_reduced 

        costbar = vmap(lambda i: jnp.where(costbar[i]>0., 1., 0.))\
            (jnp.arange(self.ellite_num_projection)) # ellite_num_projection x num_reduced 
        
        return costbar.sum(axis=1)/self.num_reduced
    
   