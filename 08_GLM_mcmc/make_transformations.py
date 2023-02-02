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
# %%
# %%
# data_mat = Path("./isConnPrediction_CV_2021_08_05.mat")


# %%
df = pd.read_csv("predictorTable.txt")

# %%
df_new = pd.DataFrame()
df_new['logBurst']=np.log(df.burstIndex)
df_new['logAdapt']=np.log(df.adaptIndex)
df_new['logRm']= np.log(df.Rm)
df_new['logCm']= np.log(df.Cm)
# df_new['logTaum']=np.log(df.Rm)+np.log(df.Cm)
df_new['logitSagRatio']=logit(df.sagRatio)
df_new['logOutputGain']=np.log(df.outputGain)
df_new['logSpikeWidth']=np.log(df.spikeWidth)

# df_new['logdML2'] = np.log(df.dML2+0.001)
# df_new['logdDV2'] = np.log(df.dDV2+0.001)
# df_new['logdAP2'] = np.log(df.dAP2+0.001)

rest_useful_columns=[
    'preML', 'preDV', 'preAP', 'postML', 'postDV', 'postAP',
    'APbregma',
    'dEuclid', 'dHoriz',
    'is_mPFCBLA2mPFCBLA', 'is_mPFCBLA2nonmPFCBLA',
    'spikeThresh', 'spikeAmp'
]
for col in rest_useful_columns:
    df_new[col]=df[col]
df_new['isConn']=df['isConn']

df_new.to_csv("transformed_dfs/logtransformed_useful.csv")
#
#  # %%
# fig,axs=plt.subplots(5,5,figsize=[20,20])
# df_new.hist(ax=axs.flatten()[:21],bins=31)
# plt.savefig("plots/beforez.png")
# plt.show()





# %%
new_columns=['logBurst', 'logAdapt', 'logRm', 'logCm', 'logitSagRatio',
       'logOutputGain', 'logSpikeWidth',
        # 'logdML2', 'logdDV2', 'logdAP2',
       'preML', 'preDV', 'preAP', 'postML', 'postDV', 'postAP', 'APbregma',
       'dEuclid', 'dHoriz', 'is_mPFCBLA2mPFCBLA', 'is_mPFCBLA2nonmPFCBLA',
       'spikeThresh', 'spikeAmp','isConn']
columns_to_zscore= ['logBurst', 'logAdapt', 'logRm', 'logCm', 'logitSagRatio',
       'logOutputGain', 'logSpikeWidth',
        # 'logdML2', 'logdDV2', 'logdAP2',
       'preML', 'preDV', 'preAP', 'postML', 'postDV', 'postAP', 'APbregma',
       'dEuclid', 'dHoriz',
       'spikeThresh', 'spikeAmp']


columns_not_zscored = [ col for col in df_new.columns if col not in columns_to_zscore]
columns_not_zscored
# %%

def my_describe(df, stats):
    d = df.describe()
    return d.append(df.reindex(d.columns, axis=1).agg(stats))

def my_zscore(df, cols):
    df_summ = my_describe(df.loc[:, cols], ['skew', 'mad', 'kurt', 'median'])
    df_zscore = pd.DataFrame()
    for col in sorted(cols):
        col_zscore = col# + '_z'
        df_zscore[col_zscore] =0.67449* (df[col] - df_summ.loc["median", col])/df_summ.loc["mad", col]
        # df_zscore[col_zscore] = (df[col] - df_summ.loc["mean", col])/df_summ.loc["std", col]        
    return df_zscore



df_newz=my_zscore(df_new,columns_to_zscore,)
for col in columns_not_zscored:
    df_newz[col]=df_new[col]


df_newz.dropna().to_csv("transformed_dfs/log_z_med_mad_valid.csv")












#%%
from sklearn.preprocessing import robust_scale

#%%
df_newz_robust=pd.DataFrame()
for col in sorted(columns_to_zscore):
    df_newz_robust[col] = robust_scale(df_new[col],unit_variance=True)
    # break
for col in columns_not_zscored:
    df_newz_robust[col] = df_new[col]

# df_newz_arr = robust_scale(df_newz)
df_newz_robust_valid=df_newz_robust.dropna()

df_newz_robust_valid.dropna().to_csv("transformed_dfs/log_z_scipy_robust_valid.csv")


#%%
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance, MinCovDet,ShrunkCovariance,OAS

#%%
df_curr =df_new.dropna() 
df_curr_toz= df_curr[sorted(columns_to_zscore)]
robust_cov = EmpiricalCovariance().fit(df_curr_toz)
# robust_cov = ShrunkCovariance().fit(df_curr_toz)
# robust_cov = OAS().fit(df_curr_toz)
# robust_cov = MinCovDet().fit(df_curr_toz)

#%%
curr_cov=robust_cov.covariance_
variances=np.diagonal(curr_cov)
stds=np.sqrt(variances)
means=robust_cov.location_

#%%
w,v = np.linalg.eigh(curr_cov)
print(np.isclose(v @  np.diagflat(w) @v.T  ,curr_cov).all())
zca_trans = v @  np.diagflat(w**(-0.5) ) @v.T
print(np.isclose(zca_trans.T @ zca_trans  ,np.linalg.inv(curr_cov)).all())


#%%
df_zca=((df_curr_toz-means) @ zca_trans)
df_zca.columns=sorted(columns_to_zscore)


#%%
corr=df_zca.corr()
corr.style.background_gradient(cmap='bwr',vmin=-1,vmax=1).set_precision(3)



#%%
fig,axs=plt.subplots(5,4,figsize=[20,20],sharex=True)
df_zca.hist(ax=axs.flatten()[:18],bins=31)
# plt.savefig("plots/beforez.png")
plt.show()

#%%
for col in columns_not_zscored:
    df_zca[col]=df_curr[col]

df_zca.dropna().to_csv("transformed_dfs/log_z_scipy_empcov_zca_valid.csv")







#########################################################
### mincovdet ZCA
#########################################################
#%%
df_curr =df_new.dropna() 
df_curr_toz= df_curr[sorted(columns_to_zscore)]
# robust_cov = EmpiricalCovariance().fit(df_curr_toz)
# robust_cov = ShrunkCovariance().fit(df_curr_toz)
# robust_cov = OAS().fit(df_curr_toz)
robust_cov = MinCovDet().fit(df_curr_toz)

#%%
curr_cov=robust_cov.covariance_
variances=np.diagonal(curr_cov)
stds=np.sqrt(variances)
means=robust_cov.location_

#%%
w,v = np.linalg.eigh(curr_cov)
print(np.isclose(v @  np.diagflat(w) @v.T  ,curr_cov).all())
zca_trans = v @  np.diagflat(w**(-0.5) ) @v.T
print(np.isclose(zca_trans.T @ zca_trans  ,np.linalg.inv(curr_cov)).all())


#%%
df_zca=((df_curr_toz-means) @ zca_trans)
df_zca.columns=sorted(columns_to_zscore)


#%%
corr=df_zca.corr()
corr.style.background_gradient(cmap='bwr',vmin=-1,vmax=1).set_precision(3)



#%%
fig,axs=plt.subplots(5,4,figsize=[20,20],sharex=True)
df_zca.hist(ax=axs.flatten()[:18],bins=31)
# plt.savefig("plots/beforez.png")
plt.show()

#%%
for col in columns_not_zscored:
    df_zca[col]=df_curr[col]

df_zca.dropna().to_csv("transformed_dfs/log_z_scipy_mincovdet_zca_valid.csv")

#%%
print("done")

#%%

