import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from EMprogram_havingsigma import AllFuncs
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.multitest import multipletests


dataTNBC = pd.read_csv('./data/GEO_tnbc.csv', index_col=0, header=0)
###### input: the dataset of TNBC
random.seed(59)
gene_sel = random.sample(list(dataTNBC.index), 1000)
label = np.where(np.array(gene_sel) == 'nan')[0]
[gene_sel.remove(cElement) for cElement in [gene_sel[i] for i in label]]
Y = pd.DataFrame(columns=dataTNBC.columns)
for s in gene_sel:
    Y.loc[s] = np.mean(dataTNBC.loc[s], axis=0)
n = Y.shape[0]
m = Y.shape[1]
gene_names = Y.index
Y = Y.reset_index(drop=True)
Y.columns = np.arange(len(Y.columns))
Inm = pd.DataFrame((np.zeros(n * m) + 1).reshape((n, m)), dtype=float)
In = np.full(n, 1)
Im = np.full(m, 1)
N_s = 4
mu_0 = np.mean(np.mean(Y))
alpha_0 = pd.DataFrame(np.mean(Y, axis=1)) - np.mean(np.mean(Y))
alpha_0 = alpha_0.reset_index(drop=True)
beta_0 = pd.DataFrame(np.mean(Y, axis=0)) - np.mean(np.mean(Y))
beta_0 = beta_0.reset_index(drop=True)
inita = pd.DataFrame(
    Y.values - mu_0 * Inm - alpha_0.dot(Im.reshape(1, m)).values - In.reshape(n, 1).dot(beta_0.transpose()))
sigma_0 = np.std(inita.values.reshape(n * m, ))
nongroup_like = -np.sum(np.sum(inita * inita)) / (2 * sigma_0 ** 2) - (n * m) * np.log(sigma_0)
y_0 = KMeans(n_clusters=N_s).fit(inita.values.reshape(-1, 1))
C_0 = list(y_0.cluster_centers_.reshape(1, N_s)[0])
P_0 = list(map(lambda k: float(len(np.where(y_0.labels_ == k)[0])) / (n * m), np.arange(N_s)))
mod = np.dot(C_0, P_0)
C_0 = C_0 - mod
c_0 = np.array(sorted(C_0))
p_0 = np.array([P_0 for _, P_0 in sorted(zip(C_0, P_0))])
estimation = AllFuncs(Y, mu_0, alpha_0, beta_0, c_0, p_0, sigma_0).EM()
beta_em = estimation["hat_beta"]
beta_hat = beta_em.values.reshape(198, )
large = np.where(beta_hat > 0)[0]
small = np.where(beta_hat < 0)[0]

Y_large = Y[large]
Y_large_std = Y_large.sub(Y_large.mean(1), axis=0).div(Y_large.std(1), axis=0)
Y_small = Y[small]
Y_samll_std = Y_small.sub(Y_large.mean(1), axis=0).div(Y_small.std(1), axis=0)

P_val_t = []
T_stat = []
for a, b in zip(Y_large.index, Y_small.index):
    Y_l_data = Y_large.iloc[a].values.reshape(len(large), 1)
    Y_s_data = Y_small.iloc[b].values.reshape(len(small), 1)
    t_stat, p_val_t = ttest_ind(Y_l_data, Y_s_data, axis=0, equal_var=False)
    P_val_t.append(p_val_t[0])
    T_stat.append(t_stat)

pd_Pval_t = pd.DataFrame(index=gene_names)
pd_Pval_t[0] = P_val_t
pd_Pval_t = pd_Pval_t.dropna()
length_t = len(pd_Pval_t)
pd_Pval_t_array = pd_Pval_t.values.reshape(length_t, )
logten_Pvalt = np.log10(pd_Pval_t_array)
upper_Pval_t = logten_Pvalt[logten_Pvalt > np.log10(0.05/length_t)]
upper_Pval_t_loc = np.where(logten_Pvalt > np.log10(0.05/length_t))[0]
lower_Pval_t = logten_Pvalt[logten_Pvalt < np.log10(0.05/length_t)]
lower_Pval_t_loc = np.where(logten_Pvalt < np.log10(0.05/length_t))[0]
Bonf_t_name = pd_Pval_t.index[lower_Pval_t_loc]

plt.title("t test P_value bonferroni criterion")
plt.scatter(upper_Pval_t_loc, upper_Pval_t, marker=".", s = 5)
plt.scatter(lower_Pval_t_loc, lower_Pval_t, marker="+", s = 5)
plt.plot(np.arange(length_t), np.log10(np.array([0.05/length_t]*length_t)), c="black")
plt.ylabel(r"$\log_{10}p$")
plt.savefig("./result/P_value_bonferroni_criterion_t.jpg")
##### output: Figure S5 in supplement
plt.show()
plt.close()

sortlogten_Pval_t = np.sort(logten_Pvalt)[1:length_t]
reset = np.arange(length_t)[1:length_t]
fdr_t = np.log10(reset/length_t * 0.05)
upper_fdr_loc_t = np.where(sortlogten_Pval_t > fdr_t)[0]
upper_Pval_fdr_t = sortlogten_Pval_t[sortlogten_Pval_t > fdr_t]
lower_fdr_loc_t = np.where(sortlogten_Pval_t < fdr_t)[0]
lower_Pval_fdr_t = sortlogten_Pval_t[sortlogten_Pval_t < fdr_t]
FDR_t_name = pd_Pval_t.index[lower_fdr_loc_t]

plt.title("t test P_value FDR criterion")
plt.scatter(upper_fdr_loc_t, upper_Pval_fdr_t, marker=".", s = 5)
plt.scatter(lower_fdr_loc_t, lower_Pval_fdr_t, marker="+", s = 5)
plt.plot(reset, np.log10(reset/length_t * 0.05), c="black")
plt.ylabel(r"$\log_{10}p$")
plt.savefig("./result/P_value_FDR_criterion_t.jpg")
##### output: Figure S5 in supplement
plt.show()
plt.close()

pd_Pval_u = pd.DataFrame(index=gene_names)
U_stat = []
for a, b in zip(Y_large.index, Y_small.index):
    Y_l_data = Y_large.iloc[a]
    Y_s_data = Y_small.iloc[b]
    if len(np.unique(Y_l_data)) == 1 & len(np.unique(Y_s_data)) == 1:
        continue
    u_stat, p_val_u = mannwhitneyu(Y_l_data, Y_s_data, alternative="two-sided")
    name = pd_Pval_u.index[a]
    pd_Pval_u.loc[name, 0] = p_val_u
    U_stat.append(u_stat)

pd_Pval_u = pd_Pval_u.dropna()
length_u = len(pd_Pval_u)
pd_Pval_u_array = pd_Pval_u.values.reshape(length_u, )
logten_Pvalu = np.log10(pd_Pval_u_array)
upper_Pval_u = logten_Pvalu[logten_Pvalu > np.log10(0.05/length_u)]
upper_Pval_u_loc = np.where(logten_Pvalu > np.log10(0.05/length_u))[0]
lower_Pval_u = logten_Pvalu[logten_Pvalu < np.log10(0.05/length_u)]
lower_Pval_u_loc = np.where(logten_Pvalu < np.log10(0.05/length_u))[0]
Bonf_u_name = pd_Pval_u.index[lower_Pval_u_loc]


plt.title("Mann Whitney U test P_value bonferroni criterion")
plt.scatter(upper_Pval_u_loc, upper_Pval_u, marker=".", s = 5)
plt.scatter(lower_Pval_u_loc, lower_Pval_u, marker="+", s = 5)
plt.plot(np.arange(length_u), np.log10(np.array([0.05/length_u]*length_u)), c="black")
plt.ylabel(r"$\log_{10}p$")
plt.savefig("./result/P_value_bonferroni_criterion_u.jpg")
#### output: Figure S5 in supplement
plt.show()
plt.close()

sortlogten_Pval_u = np.sort(logten_Pvalu)[1:length_u]
reset = np.arange(length_u)[1:length_u]
fdr_u = np.log10(reset/length_u * 0.05)
upper_fdr_loc_u = np.where(sortlogten_Pval_u > fdr_u)[0]
upper_Pval_fdr_u = sortlogten_Pval_u[sortlogten_Pval_u > fdr_u]
lower_fdr_loc_u = np.where(sortlogten_Pval_u < fdr_u)[0]
lower_Pval_fdr_u = sortlogten_Pval_u[sortlogten_Pval_u < fdr_u]
FDR_u_name = pd_Pval_u.index[lower_fdr_loc_u]

plt.title("Mann Whitney U test P_value FDR criterion")
plt.scatter(upper_fdr_loc_u, upper_Pval_fdr_u, marker=".", s = 5)
plt.scatter(lower_fdr_loc_u, lower_Pval_fdr_u, marker="+", s = 5)
plt.plot(reset, np.log10(reset/length_u * 0.05), c="black")
plt.ylabel(r"$\log_{10}p$")
plt.savefig("./result/P_value_FDR_criterion_u.jpg")
##### output: Figure S5 in supplement
plt.show()
plt.close()


