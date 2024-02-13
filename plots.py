import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import einops
import pandas as pd

import jax.numpy as jnp

import matplotlib
matplotlib.rcParams['figure.figsize'] = (13.33,5.5)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
sns.set_palette('colorblind')


def prior_predictive(model,n_prior,crosslinker_raw,crosslinker,crosslinker_macro_raw,
                     crosslinker_macro,G,G_macro,units,naming,xlabel,mec_type,key):
    init_samples = model.sample(n_prior,seed=key)
    fig,ax = plt.subplots(1,2)
    for i in range(n_prior):
        #ax[0].plot(c_pred*c_std+c_mean,tt1[i],color=sns.color_palette('colorblind')[0],zorder=0,alpha=0.2)
        l1 = ax[0].scatter(crosslinker_raw+np.random.normal(0,0.2,crosslinker.shape[0]),
                    init_samples.likelihood[i],alpha=0.1,color=sns.color_palette('colorblind')[0])
    l2 = ax[0].scatter(crosslinker_raw,G,color=sns.color_palette('colorblind')[1])
    ax[0].scatter(np.NaN,np.NaN,color=sns.color_palette('colorblind')[0],label='prior samples')
    ax[0].scatter(np.NaN,np.NaN,color=sns.color_palette('colorblind')[1],label='data')
    ax[0].legend()

    for i in range(n_prior):
        l1 = ax[1].scatter(crosslinker_macro_raw+np.random.normal(0,0.2,crosslinker_macro.shape[0]),
                    init_samples.likelihood_macro[i],alpha=0.1,color=sns.color_palette('colorblind')[0])
    l2 = ax[1].scatter(crosslinker_macro_raw,G_macro,color=sns.color_palette('colorblind')[1])

    ax[0].set_title('microrheology',fontsize=15)
    ax[1].set_title('macrorheology',fontsize=15)
    if mec_type == 'G':
        ax[0].set_ylabel(r'Normalized |G$^*$| [{}]'.format(units),fontsize=15)
        #fig.suptitle(r'Prior predictive for |G$^*$|')
    else:
        ax[0].set_ylabel(r'Normalized ${}$ [{}]'.format(naming,units))
        #fig.suptitle(r'Prior predictive for ${}$'.format(naming))
    ax[1].set_ylabel('')
    ax[0].set_xlabel(xlabel,fontsize=15)
    ax[1].set_xlabel(xlabel,fontsize=15)

    ax[0].text(0.05, 0.95, 'A', transform=ax[0].transAxes,fontsize=20)
    ax[1].text(0.05, 0.95, 'B', transform=ax[1].transAxes,fontsize=20)
    ax[0].set_xlabel(xlabel,fontsize=15)
    ax[1].set_xlabel(xlabel,fontsize=15)


    fig.savefig(f'results/prior_predictive_{mec_type}.png',bbox_inches = 'tight',dpi=600)


def plot_posterior(states,c_std,c_mean,model_type,mec_type):
    if model_type=='switchpoint':
        param_names = ['$\gamma$ (switchpoint)',
        '$\gamma_z$',
        '$\gamma^\sigma$',
        '$k^1$',
        '$k^2$',
        '$b^1$',
        '$\\alpha^\\sigma$',
        '$\\rho^\\sigma$',
        '$\\eta^\\sigma$',
        '$\sigma^{\mathrm{coating}}$',
        '$z^{\mathrm{coating}}$',
        '$\mu_{\mathrm{concentration}}$',
        '$b_{\mathrm{concentration}}$',
        '$\mu_{\mathrm{temperature}}$',
        '$b_{\mathrm{temperature}}$',
        '$\sigma^{\mathrm{holder}}$',
        '$\sigma^{\mathrm{sample}}$',
        '$z^{\mathrm{holder}}$',
        '$z^{\mathrm{sample}}$',
        '$\sigma$',
        '$\sigma^{\mathrm{probe}}$',
        '$z^{\mathrm{probe}}$',
        '$\sigma^{\mathrm{probe}\;\sigma}$',
        '$z^{\mathrm{probe}\;\sigma}$']
    elif model_type=='linear':
        param_names = [
        '$k_{\\mu}$',
        '$k_{\mathrm{intercept}}$',
        '$\\alpha^\\sigma$',
        '$\\rho^\\sigma$',
        '$\\eta^\\sigma$',
        '$\sigma^{\mathrm{coating}}$',
        '$z^{\mathrm{coating}}$',
        '$k_{\mathrm{concentration}}$',
        '$b_{\mathrm{concentration}}$',
        '$k_{\mathrm{temperature}}$',
        '$b_{\mathrm{temperature}}$',
        '$\sigma^{\mathrm{holder}}$',
        '$\sigma^{\mathrm{sample}}$',
        '$z^{\mathrm{holder}}$',
        '$z^{\mathrm{sample}}$',
        '$\sigma$',
        '$\sigma^{\mathrm{probe}}$',
        '$z^{\mathrm{probe}}$',
        '$\sigma^{\mathrm{probe}\;\sigma}$',
        '$z^{\mathrm{probe}\;\sigma}$']


    n = len(states._asdict().keys())
    n_cols = 4
    n_rows = int(np.ceil(n/n_cols))
    fig,ax = plt.subplots(n_rows,n_cols,figsize=(8.27,11.69))
    sub = 10
    def pl(ax,name,p,x_trans=None):
        d = np.copy(states._asdict()[name])
        st = 'i j -> (i j)'
        r = False
        if len(d.shape)>2:
            st = 'i j k -> (i j) k'
            r = True
        xx = einops.rearrange(d,st)
        if x_trans is not None:
            xx = x_trans(xx)
        if len(d.shape)>2:
            if d.shape[-1]>4:
                for j in range(xx.shape[-1]):
                    ax.hist(xx[::sub,j],density=True,bins=30,alpha=0.4)
            else:
                for j,k in zip(range(xx.shape[-1]),['micro alg','micro IPN','macro alg','macro IPN']):
                    ax.hist(xx[::sub,j],density=True,bins=30,alpha=0.4,label=k)
        else:
            ax.hist(xx,density=True,bins=30)
        ax.set_title(r'{}'.format(p))
        #ax.set_title(r'{}'.format(p))
        return r

    x_trans = lambda x: x*c_std+c_mean
    done = False
    for i,name,nn in zip(ax.ravel(),states._asdict(),param_names):
        if name in ['switchpoint']:
            ret = pl(i,name,nn,x_trans)
        else:
            ret = pl(i,name,nn,None)
        if ret and not done:
            done = True
            #i.legend()

    d = np.copy(states._asdict()[name])
    st = 'i j k -> (i j) k'
    xx = einops.rearrange(d,st)

    for j,k in zip(range(xx.shape[-1]),['microrheometer alginate','microrheometer IPN','macrorheometer alginate','macrorheometer IPN']):
        ax.ravel()[-2].hist([],density=True,bins=30,alpha=0.4,label=k)
    ax.ravel()[-2].legend(loc='center')
    ax.ravel()[-2].axis('off')
    ax.ravel()[-3].axis('off')
    ax.ravel()[-4].axis('off')
    ax.ravel()[-1].axis('off')
    fig.tight_layout()
    fig.savefig(f'results/all_posteriors_{mec_type}.png',bbox_inches = 'tight',dpi=600)


def posterior_comparisons(data,macro,samps,states,crosslinker_macro_raw,crosslinker_raw,G,G_macro,xlabel,naming,units,mec_type):

    sub_sample = 100
    df1 = pd.concat([pd.DataFrame({'crosslinker':crosslinker_raw,'G':G,'type':'data',
                    'material':data['type'].astype(pd.StringDtype()).reset_index()['type'],
                    'temperature':data['temp'].values,'concentration':data['concentration'].values}),
                pd.DataFrame({'G': einops.rearrange(np.array(samps.likelihood[::sub_sample]),'a b c -> (a b c)'),
                    'crosslinker':np.tile(crosslinker_raw,np.prod(states.sigma_alpha[::sub_sample].shape[:2])),
                    'type':'model','material':np.tile(data['type'].values,np.prod(states.sigma_alpha[::sub_sample].shape[:2])),
                    'temperature':np.tile(data['temp'].values,np.prod(states.sigma_alpha[::sub_sample].shape[:2])),
                    'concentration':np.tile(data['concentration'].values,np.prod(states.sigma_alpha[::sub_sample].shape[:2]))})])

    df2 = pd.concat([pd.DataFrame({'crosslinker':crosslinker_macro_raw,'G':G_macro,'type':'data',
                    'material':macro['type'].astype(pd.StringDtype()).reset_index()['type']}),
                pd.DataFrame({'G': einops.rearrange(np.array(samps.likelihood_macro[::sub_sample]),'a b c -> (a b c)'),
                    'crosslinker':np.tile(crosslinker_macro_raw,np.prod(states.sigma_alpha[::sub_sample].shape[:2])),
                    'type':'model','material':np.tile(macro['type'].values,np.prod(states.sigma_alpha[::sub_sample].shape[:2]))})])
    
    fig,ax = plt.subplots(1,2)

    sns.violinplot(data=df1[(df1['material']=='alginate') & (df1['type']=='model')],x='crosslinker',
                y='G',ax=ax[0],color=sns.color_palette('colorblind')[1])
    sns.stripplot(data=df1[(df1['material']=='alginate') & (df1['type']=='data')],
                x='crosslinker',y='G',ax=ax[0],dodge=True,color='black')

    sns.violinplot(data=df1[(df1['material']=='ipn') & (df1['type']=='model')],x='crosslinker',y='G',hue='concentration',ax=ax[1])
    sns.stripplot(data=df1[(df1['material']=='ipn') & (df1['type']=='data')],
                x='crosslinker',y='G',hue='concentration',ax=ax[1],dodge=True,palette='dark:black',legend=False)

    ax[0].set_ylabel(r'Normalized ${}$ [{}]'.format(naming,units),fontsize=15)
    ax[1].set_ylabel('')
    ax[0].set_xlabel(xlabel,fontsize=15)
    ax[1].set_xlabel(xlabel,fontsize=15)
    ax[1].set_title('ipn',fontsize=15)
    ax[0].set_title('alginate',fontsize=15)
    fig.savefig(f'results/concentration_{mec_type}.png',bbox_inches = 'tight',dpi=600)

    fig,ax = plt.subplots(1,2)

    sns.violinplot(data=df1[(df1['material']=='alginate') & (df1['type']=='model')],x='crosslinker',
                y='G',ax=ax[0],color=sns.color_palette('colorblind')[1])
    sns.stripplot(data=df1[(df1['material']=='alginate') & (df1['type']=='data')],
                x='crosslinker',y='G',ax=ax[0],dodge=True,color='black')

    sns.violinplot(data=df1[(df1['material']=='ipn') & (df1['type']=='model')],x='crosslinker',y='G',hue='temperature',ax=ax[1])
    sns.stripplot(data=df1[(df1['material']=='ipn') & (df1['type']=='data')],
                x='crosslinker',y='G',hue='temperature',ax=ax[1],dodge=True,palette='dark:black',legend=False)

    ax[0].set_ylabel(r'Normalized ${}$ [{}]'.format(naming,units),fontsize=15)
    ax[1].set_ylabel('')
    ax[0].set_xlabel(xlabel,fontsize=15)
    ax[1].set_xlabel(xlabel,fontsize=15)
    ax[1].set_title('ipn',fontsize=15)
    ax[0].set_title('alginate',fontsize=15)
    fig.savefig(f'results/temperature_{mec_type}.png',bbox_inches = 'tight',dpi=600)

    fig,ax = plt.subplots(1,2)
    sns.violinplot(data=df1[(df1['material']=='alginate') & (df1['type']=='model')],x='crosslinker',y='G',
                ax=ax[0],color=sns.color_palette('colorblind')[0])
    sns.stripplot(data=df1[(df1['material']=='alginate') & (df1['type']=='data')],
                x='crosslinker',y='G',ax=ax[0],color='black')
    sns.violinplot(data=df2[(df2['material']=='alginate') & (df2['type']=='model')],x='crosslinker',y='G',
                ax=ax[1],color=sns.color_palette('colorblind')[0])
    sns.stripplot(data=df2[(df2['material']=='alginate') & (df2['type']=='data')],
                x='crosslinker',y='G',ax=ax[1],color='black')

    ax[0].set_title('Microrheology')
    ax[1].set_title('Macrorheology')
    ax[0].set_ylabel(r'Normalized ${}$ [{}]'.format(naming,units))
    ax[1].set_ylabel('')
    ax[0].set_xlabel(xlabel)
    ax[1].set_xlabel(xlabel)
    fig.suptitle('alginate')

    fig.savefig(f'results/posterior_alginate_{mec_type}.png',bbox_inches = 'tight',dpi=600)


    fig,ax = plt.subplots(1,2)
    sns.violinplot(data=df1[(df1['material']=='ipn') & (df1['type']=='model')],x='crosslinker',y='G',
                ax=ax[0],color=sns.color_palette('colorblind')[0])
    sns.stripplot(data=df1[(df1['material']=='ipn') & (df1['type']=='data')],
                x='crosslinker',y='G',ax=ax[0],color='black')
    sns.violinplot(data=df2[(df2['material']=='ipn') & (df2['type']=='model')],x='crosslinker',y='G',
                ax=ax[1],color=sns.color_palette('colorblind')[0])
    sns.stripplot(data=df2[(df2['material']=='ipn') & (df2['type']=='data')],
                x='crosslinker',y='G',ax=ax[1],color='black')

    ax[0].set_title('Microrheology')
    ax[1].set_title('Macrorheology')
    ax[0].set_ylabel(r'Normalized ${}$ [{}]'.format(naming,units))
    ax[1].set_ylabel('')
    ax[0].set_xlabel(xlabel)
    ax[1].set_xlabel(xlabel)
    fig.suptitle('ipn')

    fig.savefig(f'results/comparison_{mec_type}.png',bbox_inches = 'tight',dpi=600)



def compare_micromacro(curve,curve_phi,c_pred,c_std,c_mean,xlabel,G_rope,phi_rope):
    def compa(x,ax,i,j,label,col,linestyle,rope,flip=False):
        if i is not None:
            t = einops.rearrange(x[:,:,i],'i j t-> (i j) t')
        if j is not None:
            tt2 = einops.rearrange(x[:,:,j],'i j t-> (i j) t')
        else:
            tt2 = einops.rearrange(x,'i j k -> (i j) k')

        if flip:
            tt = t-tt2
        else:
            tt = tt2-t
        r = (tt>rope).sum(axis=0)/tt.shape[0]
        ax.plot(c_pred*c_std+c_mean,r,color=col,label=r'{}'.format(label),linestyle=linestyle)

    cols = sns.color_palette('colorblind')
    fig,ax = plt.subplots(1,1)
    compa(curve,ax,0,2,'alginate |G*|',cols[0],'-',G_rope)
    compa(curve,ax,1,3,'ipn |G*|',cols[1],'-',G_rope)
    compa(curve_phi,ax,0,2,'alginate $\Phi$',cols[0],'--',phi_rope,True)
    compa(curve_phi,ax,1,3,'ipn $\Phi$',cols[1],'--',phi_rope,True)
    #ax.axhline(0.95,color='black')
    ax.set_xlabel(xlabel,fontsize=15)
    ax.set_ylabel('Probability',fontsize=15)
    ax.set_ylim([0,1.1])
    ax.legend()
    fig.savefig(f'results/micromacro.png',bbox_inches = 'tight',dpi=600)


def plot_mean(c_pred,c_std,c_mean,curve,sigma,N_alg,N_pred,N_ipn,N_split,N_macro_alg,
              N_macro_ipn):

    def plottos(x,ax,i=None):
        if i is not None:
            tt = einops.rearrange(x[:,:,i],'i j t-> (i j) t')
        else:
            tt = einops.rearrange(x,'i j k -> (i j) k')
        low = np.percentile(tt,5,axis=0)
        up = np.percentile(tt,95,axis=0)
        mean = np.mean(tt,axis=0)
        ax.plot(c_pred*c_std+c_mean,mean)
        ax.fill_between(c_pred*c_std+c_mean,low,up,alpha=0.1)

    def plot_posterior(x,y,ax,i):
        tt1 = einops.rearrange(x[:,:,i],'i j t-> (i j) t')
        tt2 = einops.rearrange(y,'i j k -> (i j) k')
        tt = tt1+np.random.normal(0,1,tt1.shape)*tt2
        low = np.percentile(tt,5,axis=0)
        up = np.percentile(tt,95,axis=0)
        mean = np.mean(tt,axis=0)
        ax.plot(c_pred*c_std+c_mean,mean)
        ax.fill_between(c_pred*c_std+c_mean,low,up,alpha=0.1)

    fig,ax = plt.subplots(1,3)
    plottos(curve,ax[0],0)
    plottos(curve,ax[0],1)
    plottos(sigma[:,:,N_alg:(N_alg+N_pred)],ax[1],None)
    plottos(sigma[:,:,(N_ipn+N_alg+N_pred):(N_ipn+N_alg+2*N_pred)],ax[1],None)
    plot_posterior(curve,sigma[:,:,N_alg:(N_alg+N_pred)],ax[2],0)
    plot_posterior(curve,sigma[:,:,(N_ipn+N_alg+N_pred):(N_ipn+N_alg+2*N_pred)],ax[2],1)

    ax[0].set_title('mean')
    ax[1].set_title('sigma')
    ax[2].set_title('mean+sigma')

    fig,ax = plt.subplots(1,3)
    plottos(curve,ax[0],2)
    plottos(curve,ax[0],3)
    plottos(sigma[:,:,(N_split+N_macro_alg):(N_split+N_macro_alg+N_pred)],ax[1],None)
    plottos(sigma[:,:,(N_split+N_macro_ipn+N_macro_alg+N_pred):(N_split+N_macro_ipn+N_macro_alg+2*N_pred)],ax[1],None)
    plot_posterior(curve,sigma[:,:,(N_split+N_macro_alg):(N_split+N_macro_alg+N_pred)],ax[2],2)
    plot_posterior(curve,sigma[:,:,(N_split+N_macro_ipn+N_macro_alg+N_pred):(N_split+N_macro_ipn+N_macro_alg+2*N_pred)],ax[2],3)

    ax[0].set_title('mean')
    ax[1].set_title('sigma')
    ax[2].set_title('mean+sigma')


def plot_heterogeneity(N_alg,N_ipn,N_pred,c_pred,crosslinker,type_indices,sigma,c_std,c_mean,naming,units,xlabel,mec_type):
    fig,ax = plt.subplots(1,1)
    for x1,x2,t,c,d,e,f,g in zip([N_alg,N_ipn+N_alg+N_pred],
                        [N_alg+N_pred,N_ipn+N_alg+2*N_pred],
                        [0,1],
                        sns.color_palette('colorblind')[:2],
                        ['alginate','IPN'],['-','--'],['/','\\'],
                        [np.array([5,7.5,10,20,30]),np.array([5,10,20])]):

        mask = c_pred<crosslinker[type_indices==t].max()
        res = sigma[...,x1:x2]
        c_pred_masked = c_pred[mask]
        res = einops.rearrange(res,'n m t -> (n m) t')
        res = res[...,mask]
        r_p1 = jnp.percentile(res,5,axis=0)
        r_p2 = jnp.percentile(res,95,axis=0)
        r_mean = jnp.mean(res,axis=0)

        rr = r_mean#*g_std+g_mean
        r1 = r_p1#*g_std+g_mean
        r2 = r_p2#*g_std+g_mean
        #if mec_type!='G_':
        #    rr = jnp.exp(rr)
        #    r1 = jnp.exp(r1)
        #    r2 = jnp.exp(r2)
        ax.plot(c_pred_masked*c_std+c_mean,rr,color=c,label=d,linewidth=3)
        ax.fill_between(c_pred_masked*c_std+c_mean,r1,r2,color=c,alpha=0.3)
        #ax.set_xlim(crosslinker_raw[type_indices==t].min()-0.5,crosslinker_raw[type_indices==t].max()+1)
        #ax.axvline(states.switchpoint[:,:,t].mean()*c_std+c_mean,color=c)
    #if mec_type=='G':
    #    ax.set_title(r'heterogeneity of |G$^*$|')
    #else:
    #    ax.set_title(r'heterogeneity of $\Phi$')
        c_trans = c_pred_masked*c_std+c_mean
        diffs = np.abs(g[...,None]-c_trans[None,...])
        coords = np.argmin(diffs,axis=1)
        cs = c_trans[coords]
        vals = r_mean[coords]
        ax.scatter(cs,vals,s=60)
    ax.set_xlabel(xlabel,fontsize=15)
    ax.set_ylabel(r'Heterogeneity of ${}$'.format(naming,units),fontsize=15)
    ax.tick_params(axis='y',labelsize=12)
    ax.tick_params(axis='x',labelsize=12)
    ax.text(0.05, 0.95, 'A' if mec_type=='G' else 'B', transform=ax.transAxes,fontsize=20)
    if mec_type=='G':
        ax.set_ylim([0,2.3])
    else:
        ax.set_ylim([0,1.5])
    #ax.set_ylabel('Heterogeneity')
    ax.legend()
    fig.savefig(f'results/heterogeneity_cont_{mec_type}.png',bbox_inches = 'tight',dpi=600)


def compare_heterogeneity(sigma,N_alg,N_pred,N_ipn,c_pred,c_std,c_mean,sigma_phi,xlabel,g_rope,phi_rope):
    s1 = sigma[...,N_alg:(N_alg+N_pred)]
    s2 = sigma[...,(N_ipn+N_alg+N_pred):(N_ipn+N_alg+2*N_pred)]

    ss1 = s1 
    ss2 = s2
    diff = (ss1-ss2)
    res = einops.rearrange(diff,'n m t -> (n m) t')

    s1_2 = sigma_phi[...,N_alg:(N_alg+N_pred)]
    s2_2 = sigma_phi[...,(N_ipn+N_alg+N_pred):(N_ipn+N_alg+2*N_pred)]

    ss1_2 = s1_2 
    ss2_2 = s2_2
    diff_2 = (ss1_2-ss2_2)
    res_2 = einops.rearrange(diff_2,'n m t -> (n m) t')

    r_p1 = jnp.percentile(res,5,axis=0)
    r_p2 = jnp.percentile(res,95,axis=0)

    r_mean = jnp.mean(res,axis=0)
    fig,ax = plt.subplots(1,1)
    xi = c_pred*c_std+c_mean
    ax.plot(xi,(res<-g_rope).sum(axis=0)/res.shape[0],linewidth=2,label='alginate-IPN $|G^*|$',color=sns.color_palette('colorblind')[0])
    ax.plot(xi,(res_2<-phi_rope).sum(axis=0)/res.shape[0],linewidth=2,label='alginate-IPN $\Phi$',color=sns.color_palette('colorblind')[1])
    ax.set_ylabel('Probability',fontsize=15)
    ax.set_xlabel(xlabel,fontsize=15)
    #if mec_type =='G':
    #    ax.set_title(f'Probability of alginate having smaller heterogeneity than ipn in |G$^*$|')
    #else:
    #    ax.set_title(f'Probability of alginate having smaller heterogeneity than ipn in $\Phi$')

    def vals(a,b,i):
        ax.axhspan(a,b,color=sns.color_palette('colorblind')[i],alpha=0.3)
    ax.axhline(0.95,color='black')
    #vals(0.95,0.975,1)
    #vals(0.975,0.99,2)
    #vals(0.99,0.999,3)
    #vals(0.999,1.,4)
    #ax.axhline(0.95,color=sns.color_palette('colorblind')[1],label='95% threshold',linewidth=3)
    #ax.axhline(0.975,color=sns.color_palette('colorblind')[2],label='97.5% threshold',linewidth=3)
    #ax.axhline(0.99,color=sns.color_palette('colorblind')[3],label='99% threshold',linewidth=3)
    #ax.axhline(0.999,color=sns.color_palette('colorblind')[4],label='99.9% threshold',linewidth=3)
    ax.legend()
    ax.set_xlim([5,20])
    ax.tick_params(axis='y',labelsize=12)
    ax.tick_params(axis='x',labelsize=12)
    #ax.text(0.01, 0.86, 'A', transform=ax.transAxes,fontsize=20)
    fig.savefig(f'results/heterogeneity_diff.png',bbox_inches = 'tight',dpi=600)
