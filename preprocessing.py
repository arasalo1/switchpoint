import numpy as np
import pandas as pd
import jax.numpy as jnp
import os






def load_data(path):
    data = pd.read_csv(os.path.join(path,'microdata.csv'))
    data = data[data['phi_(rad)']>np.deg2rad(0)]
    data = data.groupby(['temp','concentration','type','coating_type','size','crosslinker','day',
                        'day_repeat','sample','holder','location','track_id']).mean(numeric_only=True).reset_index(drop=False)
    data = data.dropna()
    data_raw = data.copy()
    a_mask = (np.array(data['a_(um)'])*1e3)>1
    data = data[a_mask]
    #data['G_abs'] = np.log(data['G_abs'])
    data['pre'] = 'mean'

    macro = pd.read_csv(os.path.join(path,'processed_macro.csv'),index_col=0)
    data['method'] = 'microrheometer'
    macro['method'] = 'rheometer'

    return data,macro


def generate_indices(data,macro):

    def index_macro(name):
        radius_indices = np.zeros(macro.shape[0],dtype=int)
        radius_orig = np.zeros(macro.shape[0],dtype=object)
        for idx,i in enumerate(np.unique(macro[name])):
            mask = macro[name]==i
            radius_indices[mask] = idx
            radius_orig[mask] = i
        return radius_indices,radius_orig



    macro_type_indices,macro_type_orig = index_macro('type')
    macro_cross_indices,macro_cross_orig = index_macro('crosslinker')


    def index(name):
        radius_indices = np.zeros(data.shape[0],dtype=int)
        radius_orig = np.zeros(data.shape[0],dtype=object)
        for idx,i in enumerate(np.unique(data[name])):
            mask = data[name]==i
            radius_indices[mask] = idx
            radius_orig[mask] = i
        return radius_indices,radius_orig

    radius_indices,radius_orig = index('size')
    type_indices,type_orig = index('type')
    coating_indices,coating_orig = index('coating_type')
    temp_indices,temp_orig = index('temp')
    concentration_indices,concentration_orig = index('concentration')
    cross_indices,_ = index('crosslinker')


    return macro_type_indices,radius_indices,radius_orig,\
        type_indices,type_orig,\
        coating_indices,coating_orig,\
        temp_indices,temp_orig,\
        concentration_indices,concentration_orig,\
        cross_indices



def generate_XY(data,macro,mec_type):
    crosslinker = jnp.array(data['crosslinker'])
    crosslinker_raw = crosslinker.copy()
    c_mean = crosslinker.mean()
    c_std = crosslinker.std()
    crosslinker = (crosslinker-c_mean)/c_std
    if mec_type=='G':
        G = jnp.array(data['G_abs'])
    else:
        G = jnp.array(data['phi_(rad)'])
    G_raw = G.copy()
    G = jnp.log(G)
    #G = jnp.sqrt(G)
    g_mean = G.mean()
    g_std = G.std()
    #G -= g_mean
    G = (G-g_mean)/g_std

    crosslinker_macro = jnp.array(macro['crosslinker'])
    crosslinker_macro_raw = crosslinker_macro.copy()

    crosslinker_macro = (crosslinker_macro-c_mean)/c_std
    if mec_type=='G':
        G_macro = jnp.array(macro['Complex Shear Modulus'])
    else:
        G_macro = jnp.array(np.deg2rad(macro['Phase Shift Angle'].values))
    G_raw_macro = G_macro.copy()
    G_macro = jnp.log(G_macro)

    #G -= g_mean
    G_macro = (G_macro-g_mean)/g_std

    return crosslinker,crosslinker_raw,c_mean,c_std,crosslinker_macro,crosslinker_macro_raw,\
            G,G_macro

def _gen_indices(data,names,idx,indices,orig_indices):
    # generate running indices based on the hierarchy
    
    # break recursion if at the end of hierarchy
    if idx>=len(names):
        return indices,orig_indices
    for ii,i in data.groupby(names[idx]):
        counts = i.shape[0]
        latest = 1
        l = indices[names[idx]]
        if len(l) != 0:
            latest = l[-1]+1
        # add running indices to the current level hierarchy
        indices[names[idx]].extend([latest]*counts)
        orig_indices[names[idx]].extend([ii]*counts)
        indices,orig_indices = _gen_indices(i,names,idx+1,indices,orig_indices)
    return indices,orig_indices

def generate_indices_cumulative(data,crosslinker,type_indices,crosslinker_macro,macro_type_indices,
                                radius_indices,coating_indices):
    cross_unique = np.unique(crosslinker)
    cross_unique_alg = np.unique(crosslinker[type_indices==0])
    cross_unique_ipn = np.unique(crosslinker[type_indices==1])
    N_alg = len(cross_unique_alg)
    N_ipn = len(cross_unique_ipn)
    indices = np.ones_like(crosslinker)
    for idx,i in enumerate(cross_unique):
        indices[crosslinker==i] = idx
    indices = jnp.array(indices).astype(int)

    cross_unique_macro = np.unique(crosslinker_macro)
    cross_unique_macro_alg = np.unique(crosslinker_macro[macro_type_indices==0])
    cross_unique_macro_ipn = np.unique(crosslinker_macro[macro_type_indices==1])
    N_macro = len(cross_unique_macro)

    N_macro_alg = len(cross_unique_macro_alg)
    N_macro_ipn = len(cross_unique_macro_ipn)

    indices_macro = np.ones_like(crosslinker_macro)
    for idx,i in enumerate(cross_unique_macro):
        indices_macro[crosslinker_macro==i] = idx
    indices_macro = jnp.array(indices_macro).astype(int)

    N = N_alg + N_ipn + N_macro_alg + N_macro_ipn 

    micro_coords = jnp.arange(N_alg+N_ipn,dtype=int)
    macro_coords = jnp.arange(N_alg+N_ipn,N,dtype=int)

    indices_typed = np.copy(indices+type_indices*N_alg)
    for idx,i in enumerate(np.unique(indices_typed)):
        indices_typed[indices_typed==i] = idx
    indices_typed = jnp.array(indices_typed)

    indices_typed_macro = np.copy(indices_macro+macro_type_indices*N_macro_alg)
    for idx,i in enumerate(np.unique(indices_typed_macro)):
        indices_typed_macro[indices_typed_macro==i] = idx
    indices_typed_macro = jnp.array(indices_typed_macro)

    N_radius = np.unique(radius_indices).shape[0]
    N_coating = np.unique(coating_indices).shape[0]
    N_micro = N_alg+N_ipn
    N_macro = N_macro_alg+N_macro_ipn

    micro_alg_indices = (type_indices==0).astype(int)
    micro_ipn_indices = (type_indices==1).astype(int)
    micro_mu_indices = (micro_alg_indices+micro_ipn_indices*2)-1

    macro_alg_indices = (macro_type_indices==0).astype(int)
    macro_ipn_indices = (macro_type_indices==1).astype(int)
    macro_mu_indices = (macro_alg_indices+macro_ipn_indices*2)+1



    gnames = ['temp','concentration','type','coating_type','size','crosslinker','day','day_repeat','sample','holder','location']
    g_indices = {i:[] for i in gnames}
    g_orig_indices = {i:[] for i in gnames}

    gindices,gorig_indices = _gen_indices(data,gnames,0,g_indices,g_orig_indices)


    sample_indices = jnp.array(gindices['sample'])-1
    holder_indices = jnp.array(gindices['holder'])-1

    N_dat_samples = np.max(gindices['sample'])
    N_holders = np.max(gindices['holder'])

    cross_all = jnp.concatenate([cross_unique_alg,cross_unique_ipn,cross_unique_macro_alg,cross_unique_macro_ipn])
    concentration = np.array(data['concentration'].values)
    concentration = jnp.array((concentration-concentration.mean())/concentration.std())
    temperature = np.array(data['temp'].values)
    temperature = jnp.array((temperature-temperature.min())/(temperature.max()-temperature.std()))

    types = np.zeros_like(data['type'].values,dtype=float)
    for i in data['type'].unique():
        if i=='ipn':
            types[data['type'].values==i] = 1
    types = jnp.array(types)
    G_names = []
    for i,j in zip(micro_alg_indices,micro_ipn_indices):
        if i==1:
            G_names.append('micro alginate')
        else:
            G_names.append('micro ipn')

    G_macro_names = []
    for i,j in zip(macro_alg_indices,macro_ipn_indices):
        if i==1:
            G_macro_names.append('macro alginate')
        else:
            G_macro_names.append('macro ipn')
    return cross_unique,cross_unique_alg,cross_unique_ipn,indices,cross_unique_macro,\
        cross_unique_macro_alg,cross_unique_macro_ipn,indices_macro,micro_coords,\
        macro_coords,indices_typed,indices_typed_macro,micro_alg_indices,micro_ipn_indices,\
        micro_mu_indices,macro_alg_indices,macro_ipn_indices,macro_mu_indices,N_alg,N_ipn,\
        N_macro,N_macro_alg,N_macro_ipn,N,N_radius,N_coating,N_micro,N_macro,sample_indices,\
        holder_indices,N_dat_samples,N_holders,cross_all,concentration,temperature,types,G_names,G_macro_names