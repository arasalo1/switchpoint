import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
import einops
from functools import partial

import pandas as pd
import numpy as np
import arviz as az
import arviz.labels as azl

from tensorflow_probability.substrates import jax as tfp
tfd =  tfp.distributions
tfb = tfp.bijectors


def rbf(x,alpha,rho):
    diff = x[None,...]-x[:,None,:]
    dist = jnp.linalg.norm(diff,axis=-1)**2
    cov = alpha**2*jnp.exp(-dist/(2*rho**2))
    return cov[...,None]


sigmoid = lambda x,sat: 1/(1+jnp.exp(-sat*x))


def gen_model(crosslinker,concentration,temperature,cross_all,c_mean,c_std,N,N_coating,N_holders,N_dat_samples,N_radius,
              coating_indices,holder_indices,sample_indices,radius_indices,cross_unique_alg,cross_unique_ipn,
              cross_unique_macro_alg,cross_unique_macro_ipn,micro_coords,indices_typed,micro_mu_indices,types,
              macro_coords,indices_typed_macro,macro_mu_indices,model_type):

    if model_type=='linear':
        @tfd.JointDistributionCoroutineAutoBatched
        def model():

            slope1 = yield tfd.Sample(tfd.Normal(0.,1.),(1,4),name='slope1')
            intercept = yield tfd.Sample(tfd.Normal(0.,1.),(1,4),name='intercept')


            #saturation = yield tfd.Sample(tfd.HalfNormal(1.),(4,),name='saturation')
            #mu = yield tfd.Sample(tfd.Normal(0.,1.),(4,),name='mu')
            #inter = yield tfd.Sample(tfd.Normal(0.,1.),(4,),name='inter')
            
            sigma_alpha = yield tfd.Sample(tfd.HalfNormal(1),(4,),name='sigma_alpha')
            sigma_rho = yield tfd.Sample(tfd.InverseGaussian(1.,3.),(4,),name='sigma_rho')
            sigma_eta = yield tfd.Sample(tfd.Normal(0.,1.),(N,),name='sigma_eta')

            #coating_mu = yield tfd.Normal(0,1,name='coating_mu')
            coating_std = yield tfd.HalfNormal(1,name='coating_std')
            coating_z = yield tfd.Normal([0]*N_coating,1,name='coating_z')
            coating_effects = jnp.ones_like(crosslinker)*(coating_std*coating_z[coating_indices])

            # concentration
            concentration_mean = yield tfd.Normal(0,1,name='concentration_mu')
            concentration_intercept = yield tfd.Normal(0,1,name='concentration_intercept')
            concentration_effect = concentration_mean*concentration+concentration_intercept

            # temperature
            temperature_mean = yield tfd.Normal(0,1,name='temperature_mu')
            temperature_intercept = yield tfd.Normal(0,1,name='temperature_intercept')
            temperature_effect = temperature_mean*temperature+temperature_intercept

            holder_std = yield tfd.HalfNormal(1.,name='holder_std')
            sample_std = yield tfd.HalfNormal(1.,name='sample_std')
            holder_z = yield tfd.Normal([0]*N_holders,1,name='holder_z')
            sample_z = yield tfd.Normal([0]*N_dat_samples,1,name='sample_z')
            #coating_sigma_std = yield tfd.HalfNormal(1,name='coating_sigma_std')
            #coating_sigma_z = yield tfd.Normal([0]*N_coating,1,name='coating_sigma_z')
            #coating_sigma_effects = jnp.ones_like(crosslinker)*(coating_sigma_std*coating_sigma_z[coating_indices])
            holder_effect = jnp.ones_like(crosslinker)*(holder_std*holder_z[holder_indices])
            sample_effect = jnp.ones_like(crosslinker)*(sample_std*sample_z[sample_indices])
            #mean_mu = yield tfd.Sample(tfd.Normal(0,1),(4,),name='mean_mu')
            sigma_mu = yield tfd.Sample(tfd.Normal(0,1),(4,),name='sigma_mu')
            #radius_mu = yield tfd.Normal(0,1,name='radius_mu')
            radius_std = yield tfd.HalfNormal(1,name='radius_std')
            radius_z = yield tfd.Normal([0]*N_radius,1,name='radius_z')
            radius_effects = jnp.ones_like(crosslinker)*(radius_std*radius_z[radius_indices])

            #radius_sigma_mu = yield tfd.Normal(0,1,name='radius_sigma_mu')
            radius_sigma_std = yield tfd.HalfNormal(1,name='radius_sigma_std')
            radius_sigma_z = yield tfd.Normal([0]*N_radius,1,name='radius_sigma_z')
            radius_sigma_effects = jnp.ones_like(crosslinker)*(radius_sigma_std*radius_sigma_z[radius_indices])
            
            K_sigma = jax.scipy.linalg.block_diag(
                rbf(cross_unique_alg[...,None],sigma_alpha[0],sigma_rho[0])[...,0],
                rbf(cross_unique_ipn[...,None],sigma_alpha[1],sigma_rho[1])[...,0],
                rbf(cross_unique_macro_alg[...,None],sigma_alpha[2],sigma_rho[2])[...,0],
                rbf(cross_unique_macro_ipn[...,None],sigma_alpha[3],sigma_rho[3])[...,0])+jnp.eye(N)*(1e-5)
            L_K_sigma = jnp.linalg.cholesky(K_sigma)

            curve = slope1*cross_all[...,None]+intercept
            #curve = saturation*sigmoid((cross_all-cross_all.min())[...,None],mu)+inter
            sigma = L_K_sigma@sigma_eta
            #+holder_effect+sample_effect+\
            lik = yield tfd.Normal(loc=curve[micro_coords][indices_typed,micro_mu_indices]+radius_effects+coating_effects+holder_effect+sample_effect+\
                                            concentration_effect*types+temperature_effect*types,
                                        scale=jnn.softplus(sigma[micro_coords][indices_typed]+sigma_mu[micro_mu_indices]+radius_sigma_effects),
                                        name='likelihood')
            lik_mac = yield tfd.Normal(loc=curve[macro_coords][indices_typed_macro,macro_mu_indices],
                                    scale=jnn.softplus(sigma[macro_coords][indices_typed_macro]+sigma_mu[macro_mu_indices]),name='likelihood_macro')
    
    elif model_type=='switchpoint':
        @tfd.JointDistributionCoroutineAutoBatched
        def model():

            #switchpoint = yield tfd.Sample(tfd.Normal((20-c_mean)/c_std,2.),(1,2),name='switchpoint')
            switchpoint = yield tfd.Normal((20-c_mean)/c_std,2.,name='switchpoint')
            slope1 = yield tfd.Sample(tfd.Normal(0.,1.),(1,4),name='slope1')
            slope2 = yield tfd.Sample(tfd.Normal(0.,1.),(1,4),name='slope2')
            intercept = yield tfd.Sample(tfd.Normal(0.,1.),(1,4),name='intercept')

            #switch = switchpoint[...,[0,0,1,1]]
            switch = switchpoint
            #switch = tfb.Shift((20-c_mean)/c_std).forward(switchpoint)[...,[0,0,1,1]]
            i2 = (slope2*switch+intercept)-slope1*switch
            slope = slope2+sigmoid(switch-cross_all[...,None],10)*(slope1-slope2)
            inter = intercept+sigmoid(switch-cross_all[...,None],10)*(i2-intercept)
            
            sigma_alpha = yield tfd.Sample(tfd.HalfNormal(1),(4,),name='sigma_alpha')
            sigma_rho = yield tfd.Sample(tfd.InverseGaussian(1.,3.),(4,),name='sigma_rho')
            sigma_eta = yield tfd.Sample(tfd.Normal(0.,1.),(N,),name='sigma_eta')

            #coating_mu = yield tfd.Normal(0,1,name='coating_mu')
            coating_std = yield tfd.HalfNormal(1,name='coating_std')
            coating_z = yield tfd.Normal([0]*N_coating,1,name='coating_z')
            coating_effects = jnp.ones_like(crosslinker)*(coating_std*coating_z[coating_indices])

            # concentration
            concentration_mean = yield tfd.Normal(0,1,name='concentration_mu')
            concentration_intercept = yield tfd.Normal(0,1,name='concentration_intercept')
            concentration_effect = concentration_mean*concentration+concentration_intercept

            # temperature
            temperature_mean = yield tfd.Normal(0,1,name='temperature_mu')
            temperature_intercept = yield tfd.Normal(0,1,name='temperature_intercept')
            temperature_effect = temperature_mean*temperature+temperature_intercept

            holder_std = yield tfd.HalfNormal(1.,name='holder_std')
            sample_std = yield tfd.HalfNormal(1.,name='sample_std')
            holder_z = yield tfd.Normal([0]*N_holders,1,name='holder_z')
            sample_z = yield tfd.Normal([0]*N_dat_samples,1,name='sample_z')

            holder_effect = jnp.ones_like(crosslinker)*(holder_std*holder_z[holder_indices])
            sample_effect = jnp.ones_like(crosslinker)*(sample_std*sample_z[sample_indices])
            #coating_sigma_std = yield tfd.HalfNormal(1,name='coating_sigma_std')
            #coating_sigma_z = yield tfd.Normal([0]*N_coating,1,name='coating_sigma_z')
            #coating_sigma_effects = jnp.ones_like(crosslinker)*(coating_sigma_std*coating_sigma_z[coating_indices])

            #mean_mu = yield tfd.Sample(tfd.Normal(0,1),(4,),name='mean_mu')
            sigma_mu = yield tfd.Sample(tfd.Normal(0,1),(4,),name='sigma_mu')
            #radius_mu = yield tfd.Normal(0,1,name='radius_mu')
            radius_std = yield tfd.HalfNormal(1,name='radius_std')
            radius_z = yield tfd.Normal([0]*N_radius,1,name='radius_z')
            radius_effects = jnp.ones_like(crosslinker)*(radius_std*radius_z[radius_indices])

            #radius_sigma_mu = yield tfd.Normal(0,1,name='radius_sigma_mu')
            radius_sigma_std = yield tfd.HalfNormal(1,name='radius_sigma_std')
            radius_sigma_z = yield tfd.Normal([0]*N_radius,1,name='radius_sigma_z')
            radius_sigma_effects = jnp.ones_like(crosslinker)*(radius_sigma_std*radius_sigma_z[radius_indices])
            
            K_sigma = jax.scipy.linalg.block_diag(
                rbf(cross_unique_alg[...,None],sigma_alpha[0],sigma_rho[0])[...,0],
                rbf(cross_unique_ipn[...,None],sigma_alpha[1],sigma_rho[1])[...,0],
                rbf(cross_unique_macro_alg[...,None],sigma_alpha[2],sigma_rho[2])[...,0],
                rbf(cross_unique_macro_ipn[...,None],sigma_alpha[3],sigma_rho[3])[...,0])+jnp.eye(N)*(1e-5)
            L_K_sigma = jnp.linalg.cholesky(K_sigma)

            curve = slope*cross_all[...,None]+inter
            sigma = L_K_sigma@sigma_eta
            
            lik = yield tfd.Normal(loc=curve[micro_coords][indices_typed,micro_mu_indices]+radius_effects+coating_effects+\
                                        holder_effect+sample_effect+\
                                        concentration_effect*types+temperature_effect*types,
                                        scale=jnn.softplus(sigma[micro_coords][indices_typed]+radius_sigma_effects+sigma_mu[micro_mu_indices]),
                                        name='likelihood')
            lik_mac = yield tfd.Normal(loc=curve[macro_coords][indices_typed_macro,macro_mu_indices],
                                    scale=jnn.softplus(sigma[macro_coords][indices_typed_macro]+sigma_mu[macro_mu_indices]),name='likelihood_macro')
    return model

def run_nuts(init_samples,target,bijector,model,key,num_chains = 4,num_steps= 5000,num_adaptation = 1000,num_burnin=1000,step_size = 1.):

    step_sizes = [jnp.ones((num_chains,*i.shape))*step_size for i in init_samples[:-2]]

    @jax.jit
    def run_chain(key, state):

        hmc = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=target.unnormalized_log_prob,
            step_size=step_sizes)

        hmc = tfp.mcmc.TransformedTransitionKernel(
            hmc, bijector)
        hmc = tfp.mcmc.DualAveragingStepSizeAdaptation(
            hmc,
            num_adaptation_steps=int(num_adaptation),
            target_accept_prob=0.9,
            reduce_fn=tfp.math.reduce_log_harmonic_mean_exp)

        return tfp.mcmc.sample_chain(num_steps,
        current_state=state,
        kernel=hmc,
        trace_fn=lambda _, results: results.inner_results.inner_results,
        seed=key)


    states_, log_probs_ = run_chain(key,jax.tree_map(lambda x: jnp.ones_like(x),model.sample(num_chains,seed=jr.split(key)[0])[:-2]))
    log_probs = jax.tree_map(lambda x: x[num_burnin:,:],log_probs_)
    states = jax.tree_map(lambda x: x[num_burnin:,:],states_)
    return states,log_probs

def gen_predictive(model,states):
    def gen_samples(params):
        dists, samps = model.sample_distributions(seed=jr.PRNGKey(0),
                                            value=params + (None,))
        return samps
        
    samps = jax.vmap(jax.vmap(gen_samples))(states)

    def gen_dists(params,G):
        dists, samps = model.sample_distributions(seed=jr.PRNGKey(0),
                                            value=params + (None,))
        return dists[-2].prob(G)
    
    #dists = jax.vmap(jax.vmap(gen_dists,in_axes=(0,None)),in_axes=(0,None))(states,G[:10][...,None]).mean(axis=2)
    return samps


def diagnostics(states,log_probs,samps,radius_orig,coating_orig,G_names,G_macro_names,G,G_macro,model_type):

    coords = {
        'radius': np.unique(radius_orig),
        'coating': np.unique(coating_orig),
        'names': ['micro alginate','micro IPN','macro alginate','macro IPN'],
        'names2':['micro','macro'],
        'G_names':G_names,
        'G_macro_names':G_macro_names,
    }
    if model_type=='switchpoint':
        dims = {
            #'switchpoint': ['names2'],
            'slope1': ['names'],
            'slope2': ['names'],
            'intecept':['names'],
            #'eta':
            'sigma_alpha': ['names'],
            'sigma_rho': ['names'],
            #'mean_mu':['names'],
            'sigma_mu':['names'],
            #'sigma_eta': 
            #'coating_std': ['dim'],
            'coating_z': ['coating'],
            #'coating_sigma_z': ['coating'],
            #'radius_std': ['dim'],
            'radius_z': ['radius'],
            #'radius_sigma_std': ['dim'],
            'radius_sigma_z': ['radius'],
            'likelihood':['G_names'],
            'likelihood_macro':['G_macro_names'],
        }
    else:
        dims = {
            #'slope1': ['names'],
            #'intecept':['names'],
            'mu':['names'],
            'inter':['names'],
            'saturation':['names'],
            #'eta':
            'sigma_alpha': ['names'],
            'sigma_rho': ['names'],
            #'mean_mu':['names'],
            'sigma_mu':['names'],
            #'sigma_eta': 
            #'coating_std': ['dim'],
            'coating_z': ['coating'],
            #'coating_sigma_z': ['coating'],
            #'radius_std': ['dim'],
            'radius_z': ['radius'],
            #'radius_sigma_std': ['dim'],
            'radius_sigma_z': ['radius'],
        }

    trace = az.from_dict(
    posterior=jax.tree_map(lambda x: jnp.swapaxes(x,0,1),states._asdict()),
    posterior_predictive=jax.tree_map(lambda x: jnp.swapaxes(x,0,1),samps._asdict()),
    #prior=jax.tree_map(lambda x: x[None,...],init_samples._asdict()),
    observed_data={'likelihood':G,'likelihood_macro':G_macro},
    #sample_stats={'log_likelihood':jnp.swapaxes(log_probs_.target_log_prob,0,1)},
    sample_stats={'log_likelihood':jnp.swapaxes(log_probs.target_log_prob,0,1),
                    'energy':jnp.swapaxes(log_probs.energy,0,1),
                    'diverging':jnp.swapaxes(log_probs.has_divergence,0,1)},
    coords=coords,
    dims=dims
    )
    trace.posterior['radius_effect'] = trace.posterior['radius_z']*trace.posterior['radius_std']
    trace.posterior['radius_sigma_effect'] = trace.posterior['radius_sigma_z']*trace.posterior['radius_sigma_std']
    trace.posterior['coating_effect'] = trace.posterior['coating_z']*trace.posterior['coating_std']
    trace.posterior['holder_effect'] = trace.posterior['holder_z']*trace.posterior['holder_std']#+trace.posterior['holder_mu']
    trace.posterior['sample_effect'] = trace.posterior['sample_z']*trace.posterior['sample_std']#+trace.posterior['sample_mu']
    #trace.posterior['coating_sigma_effect'] = trace.posterior['coating_sigma_z']*trace.posterior['coating_sigma_std']

    return trace


def mean_predictions(states,cross_unique_alg,cross_unique_ipn,cross_unique_macro_alg,cross_unique_macro_ipn,cross_unique,
                     N_pred,N_alg,N_ipn,N_micro,N_macro_alg,N_macro_ipn,c_pred,m2_,key,model_type):

    eta_key,sigma_eta_key = jr.split(key,2)
    #N_samples = states.eta.shape[0]
    num_c = states.sigma_alpha.shape[1]
    sub_sample = 3
    N_samples = states.sigma_rho[::sub_sample].shape[0]

    combined_alg = jnp.concatenate([cross_unique_alg,c_pred])
    combined_ipn = jnp.concatenate([cross_unique_ipn,c_pred])

    combined_macro_alg = jnp.concatenate([cross_unique_macro_alg,c_pred])
    combined_macro_ipn = jnp.concatenate([cross_unique_macro_ipn,c_pred])

    N_combined_alg = combined_alg.shape[0]
    N_combined_ipn = combined_ipn.shape[0]

    N_split = N_combined_alg+N_combined_ipn

    N_combined_macro_alg = combined_macro_alg.shape[0]
    N_combined_macro_ipn = combined_macro_ipn.shape[0]

    N_both = N_combined_alg+N_combined_ipn+N_combined_macro_alg+N_combined_macro_ipn

    if model_type=='switchpoint':
        slope1 = states.slope1[::sub_sample,:][...,None]
        slope2 = states.slope2[::sub_sample,:][...,None]
        #switchpoint = states.switchpoint[::sub_sample,:,:][...,None]
        switchpoint = states.switchpoint[::sub_sample,:][...,None]
        intercept = states.intercept[::sub_sample,:][...,None]
        switch = switchpoint[...,None]
        #switch = switchpoint[:,:,[0,0,1,1]]
        #switch = tfb.Shift((20-c_mean)/c_std).forward(switchpoint)[:,:,[0,0,1,1]]
        i2 = (slope2*switch+intercept)-slope1*switch
        slope = slope2+sigmoid(switch-c_pred[None,None,None,...],10)*(slope1-slope2)
        inter = intercept+sigmoid(switch-c_pred[None,None,None,...],10)*(i2-intercept)
        curve = slope*c_pred[None,None,None,...]+inter

        slope_datapoints = slope2+sigmoid(switch-cross_unique[None,None,None,...],10)*(slope1-slope2)
        inter_datapoints = intercept+sigmoid(switch-cross_unique[None,None,None,...],10)*(i2-intercept)
        curve_datapoints = slope_datapoints*cross_unique[None,None,None,...]+inter_datapoints
    else:
        slope1 = states.slope1[::sub_sample,:][...,None]
        intercept = states.intercept[::sub_sample,:][...,None]
        curve = slope1*c_pred[None,None,None,...]+intercept
        curve_datapoints = slope1*cross_unique[None,None,None,...]+intercept


    eta_pred_sigma = jr.normal(sigma_eta_key,(*states.sigma_eta.shape[:2],N_pred))
    eta_both_sigma = jnp.dstack([states.sigma_eta[...,:N_alg],eta_pred_sigma,
                                states.sigma_eta[...,N_alg:(N_alg+N_ipn)],eta_pred_sigma,
                                states.sigma_eta[...,N_micro:(N_micro+N_macro_alg)],eta_pred_sigma,
                                states.sigma_eta[...,(N_micro+N_macro_alg):((N_micro+N_macro_alg+N_macro_ipn))],eta_pred_sigma])


    N_total_sigma = eta_both_sigma.shape[-1]

    K_sigma = jax.vmap(jax.vmap(jax.scipy.linalg.block_diag))(
                    # micro
                    einops.rearrange(jax.vmap(partial(rbf,combined_alg[...,None]))(
                    einops.rearrange(states.sigma_alpha[::sub_sample,:,0],'n c -> (n c)'),
                    einops.rearrange(states.sigma_rho[::sub_sample,:,0],'n c -> (n c)'))[...,0],
                    '(n c) a b -> n c a b',n=N_samples,c=num_c),

                    einops.rearrange(jax.vmap(partial(rbf,combined_ipn[...,None]))(
                    einops.rearrange(states.sigma_alpha[::sub_sample,:,1],'n c -> (n c)'),
                    einops.rearrange(states.sigma_rho[::sub_sample,:,1],'n c -> (n c)'))[...,0],
                    '(n c) a b -> n c a b',n=N_samples,c=num_c),
                    # macro
                    einops.rearrange(jax.vmap(partial(rbf,combined_macro_alg[...,None]))(
                    einops.rearrange(states.sigma_alpha[::sub_sample,:,2],'n c -> (n c)'),
                    einops.rearrange(states.sigma_rho[::sub_sample,:,2],'n c -> (n c)'))[...,0],
                    '(n c) a b -> n c a b',n=N_samples,c=num_c),
                    
                    einops.rearrange(jax.vmap(partial(rbf,combined_macro_ipn[...,None]))(
                    einops.rearrange(states.sigma_alpha[::sub_sample,:,3],'n c -> (n c)'),
                    einops.rearrange(states.sigma_rho[::sub_sample,:,3],'n c -> (n c)'))[...,0],
                    '(n c) a b -> n c a b',n=N_samples,c=num_c)) + jnp.eye(N_both)[None,None,...]*(1e-4)

    L_K_sigma = jnp.linalg.cholesky(K_sigma)

    sigma = jnn.softplus(jnp.einsum('nmij,nmj->nmi',L_K_sigma,eta_both_sigma[::sub_sample])+states.sigma_mu[::sub_sample,:,m2_])
    curve_datapoints = einops.rearrange(curve_datapoints,'i j k n -> (i j) k n')
    return curve,curve_datapoints,sigma,N_split


def increase_in_heterogeneity(states,cross_unique_alg,cross_unique_ipn,N_micro,c_std,
                              c_mean,N_alg,N_ipn,cross_unique_macro_alg,cross_unique_macro_ipn,
                              N_macro,N_macro_alg,N_macro_ipn,m1,m2):

    sub_sample = 1
    N_samples = states.sigma_rho[::sub_sample].shape[0] 
    num_c = states.sigma_alpha.shape[1]
    K_sigma = jax.vmap(jax.vmap(jax.scipy.linalg.block_diag))(einops.rearrange(jax.vmap(partial(rbf,cross_unique_alg[...,None]))(
                    einops.rearrange(states.sigma_alpha[::sub_sample,:,0],'n c -> (n c)'),
                    einops.rearrange(states.sigma_rho[::sub_sample,:,0],'n c -> (n c)'))[...,0],
                    '(n c) a b -> n c a b',n=N_samples,c=num_c),
            einops.rearrange(jax.vmap(partial(rbf,cross_unique_ipn[...,None]))(
                    einops.rearrange(states.sigma_alpha[::sub_sample,:,1],'n c -> (n c)'),
                    einops.rearrange(states.sigma_rho[::sub_sample,:,1],'n c -> (n c)'))[...,0],
                    '(n c) a b -> n c a b',n=N_samples,c=num_c)) + jnp.eye(N_micro)[None,None,...]*(1e-5)


    L_K_sigma = jnp.linalg.cholesky(K_sigma)
    sigma = einops.rearrange(jnn.softplus(jnp.einsum('nmij,nmj->nmi',L_K_sigma,states.sigma_eta[::sub_sample,:,:N_micro])+states.sigma_mu[::sub_sample,:,m1]),
                            'a b c -> (a b) c')

    df2 = pd.DataFrame({'crosslinker':np.tile(np.round(np.concatenate([cross_unique_alg,cross_unique_ipn])*c_std+c_mean,1),
                                            sigma.shape[0]),'heterogeneity':sigma.flatten(),
                        'type':np.tile([*['alg']*N_alg,*['ipn']*N_ipn],sigma.shape[0])})

    K_sigma = jax.vmap(jax.vmap(jax.scipy.linalg.block_diag))(einops.rearrange(jax.vmap(partial(rbf,cross_unique_macro_alg[...,None]))(
                    einops.rearrange(states.sigma_alpha[::sub_sample,:,2],'n c -> (n c)'),
                    einops.rearrange(states.sigma_rho[::sub_sample,:,2],'n c -> (n c)'))[...,0],
                    '(n c) a b -> n c a b',n=N_samples,c=num_c),
            einops.rearrange(jax.vmap(partial(rbf,cross_unique_macro_ipn[...,None]))(
                    einops.rearrange(states.sigma_alpha[::sub_sample,:,3],'n c -> (n c)'),
                    einops.rearrange(states.sigma_rho[::sub_sample,:,3],'n c -> (n c)'))[...,0],
                    '(n c) a b -> n c a b',n=N_samples,c=num_c)) + jnp.eye(N_macro)[None,None,...]*(1e-5)


    L_K_sigma = jnp.linalg.cholesky(K_sigma)
    sigma = einops.rearrange(jnn.softplus(jnp.einsum('nmij,nmj->nmi',L_K_sigma,states.sigma_eta[::sub_sample,:,N_micro:])+states.sigma_mu[::sub_sample,:,m2]),
                            'a b c -> (a b) c')

    df3 = pd.DataFrame({'crosslinker':np.tile(np.round(np.concatenate([cross_unique_macro_alg,cross_unique_macro_ipn])*c_std+c_mean,1),
                                            sigma.shape[0]),'heterogeneity':sigma.flatten(),
                        'type':np.tile([*['alg']*N_macro_alg,*['ipn']*N_macro_ipn],sigma.shape[0])})

    df2.loc[df2['type']=='alg','type'] = 'alginate'

    di = []
    naming = []
    for t,i in df2.groupby('type'):
        it = 0
        for idx,j in i.groupby('crosslinker'):
            it += 1
            for ii,(idx2,k) in enumerate(i.groupby('crosslinker')):
                if ii<it:
                    continue
                a = idx
                b = idx2
                if idx2<idx:
                    a = idx2
                    b = idx
                    di.append(j['heterogeneity'].values-k['heterogeneity'].values)
                else:
                    di.append(k['heterogeneity'].values-j['heterogeneity'].values)
                naming.append('{} {} {}'.format(t,a,b))

    df5 = pd.DataFrame(np.stack(di).T,columns=naming)
    df5 = df5.melt()
    df5[['type','left','right']] = df5.variable.str.split(" ", expand = True)

    prob_df = []
    for i,j in df5.groupby('variable'):
        l = float(j['left'].values[0])
        r = float(j['right'].values[0])
        vals = j['value'].values
        out = []
        if r>l:
            comp = (vals>0.1).sum()/vals.shape[0]*100
            print(i,np.round(comp,2),end=' ')
        else:
            comp = (vals<-0.1).sum()/vals.shape[0]*100
            print(i,np.round(comp,2),end=' ')
        out.extend([i,l,r,comp])
        if comp>95:
            print('*',end='')
            out.append(True)
            if comp>97.5:
                print('*',end='')
            if comp>99:
                print('*',end='')
            if comp>99.9:
                print('*',end='')
            print(' ')
        else:
            print(' ')
            out.append(False)
        prob_df.append(out)
    prob_df = pd.DataFrame(np.array(prob_df),columns=['type','material1','material2','probability','decision'])
    #prob_df.columns = ['material1','material2','probability','decision']