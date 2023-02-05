
# %%import numpy as np
from numpy.core.numeric import ones
import scipy
from scipy import io
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import pymc3 as pm
import seaborn
import arviz as az
import numpy as np
import theano.tensor as tt
import theano
from sklearn.preprocessing import PowerTransformer
from scipy.special import logit,expit


from pathlib import Path
# %%
df_folder = Path("transformed_dfs")
plots_folder=Path("plots2")

dfs=list(df_folder.glob("log_z*mincovdet*.csv"))
#%%
df_name=dfs[0]
print(df_name)
# colum
#%%
for df_name in dfs:
    print(df_name)
    df_curr=pd.read_csv(df_name,index_col=0)
    X = df_curr.drop('isConn',axis=1).to_numpy()
    Y = df_curr['isConn'].to_numpy()  # [:1000]
    coords = {"columns": list(df_curr.drop('isConn',axis=1).columns)}
    print(df_curr.shape)
    with pm.Model(coords=coords) as logreg:
        ν = 3
        rₗ = pm.Normal('r_local', mu=0, sd=1., dims="columns")
        ρₗ = pm.InverseGamma('rho_local', alpha=0.5*ν, beta=0.5*ν, dims="columns", testval=0.1)
        rᵧ = pm.Normal('r_global', mu=0, sd=1.0) 
        ρᵧ = pm.InverseGamma('rho_global', alpha=0.5, beta=0.5, testval=0.1)
        τ = rᵧ * pm.math.sqrt(ρᵧ)
        λ = rₗ * pm.math.sqrt(ρₗ)
        z = pm. Normal('z', mu=0, sd=1, dims="columns")
        beta = pm.Deterministic('beta', z*λ*τ, dims="columns")
        beta_0 = pm.Normal('beta_0', mu=-3.678, sd=1, testval=-3)
        # trace_prior = pm.sample(2000, target_accept=.95, return_inferencedata=True,
                        # init="jitter+adapt_diag", cores=1)
        # mu = pm.math.invlogit(tt.dot(X, beta)+beta_0)
        mu_logit = tt.dot(X, beta)+beta_0
        y_obs = pm.Bernoulli('obs', logit_p=mu_logit, observed=Y)
        # prior = pm.sample_prior_predictive(100)

    with logreg:
        trace1 = pm.sample(2000, target_accept=.95, tune=10000, return_inferencedata=True,
                            init="advi+adapt_diag",
                            #  init="jitter+adapt_diag",
                            n_init=50000,
                            cores=1
                            )

    az.plot_forest(trace1, var_names=['beta'], hdi_prob=0.90, rope=[-0.05, 0.05], quartiles=True, combined=True)
    plt.savefig(plots_folder / (df_name.stem + "forest.png"))
    plt.savefig(plots_folder / (df_name.stem + "forest.svg"))
    plt.show()


    fig,axs=plt.subplots(5,4,sharex=True,figsize=[15,20],sharey=True)
    az.plot_posterior(trace1, var_names=['beta'], hdi_prob=0.90,ax=axs)
    for ax in axs.flatten():
        ax.axvline(0.0,color='black')
    fig.tight_layout()
    plt.savefig(plots_folder / (df_name.stem + "density.png"))
    plt.savefig(plots_folder / (df_name.stem + "density.svg"))
    plt.show()
# %%




# with pm.Model(coords=coords) as logreg1:
#     lam = pm.HalfCauchy('lambda', beta=1, dims="columns")
#     lamb = pm.Deterministic('log_lambda', pm.math.log(lam), dims="columns")
#     tau = pm.HalfCauchy('tau', beta=0.2)
#     sigma = pm.Deterministic('horseshoe', tau*lam, dims="columns")
#     beta_tilde = pm.Normal('beta_tilde', mu=0, sigma=1, dims="columns", testval=0.1)
#     beta = pm.Deterministic('beta', sigma*beta_tilde, dims="columns")

#     beta_0 = pm.Normal('beta_0', mu=-3.678, sd=1, testval=-3.678)
#     mu = pm.math.invlogit(tt.dot(X, beta)+beta_0)
#     y_obs = pm.Bernoulli('obs', p=mu, observed=Y)


# #%%
# with logreg1:
#     trace1 = pm.sample(1000, target_accept=.95, tune=3000, return_inferencedata=True,
#                          init="advi+adapt_diag",
#                          #  init="jitter+adapt_diag",
#                          n_init=50000,
#                          cores=1
#                          )
# #%%
# # plt.savefig("all_params_non_centered.png")

# # %%
# az.plot_forest(trace1, var_names=['horseshoe'], hdi_prob=0.5, rope=[0.01, 0.1], quartiles=False, combined=True)

# # %%
