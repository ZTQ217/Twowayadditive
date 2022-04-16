import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from EMprogram_havingsigma import AllFuncs
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import fdrcorrection

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
A_ijk = estimation["A_ijk"]
A_1 = A_ijk[0]
A_4 = A_ijk[3]
A_2 = A_ijk[1]
A_3 = A_ijk[2]
beta_hat = beta_em.values.reshape(198, )
large = np.where(beta_hat > 0)[0]
small = np.where(beta_hat < 0)[0]
A_1_large = A_1[large]
A_1_small = A_1[small]
n_1_large = A_1_large.sum(axis=1)
n_1_small = A_1_small.sum(axis=1)
A_2_large = A_2[large]
A_2_small = A_2[small]
n_2_large = A_2_large.sum(axis=1)
n_2_small = A_2_small.sum(axis=1)
A_3_large = A_3[large]
A_3_small = A_3[small]
n_3_large = A_3_large.sum(axis=1)
n_3_small = A_3_small.sum(axis=1)
A_4_large = A_4[large]
A_4_small = A_4[small]
n_4_large = A_4_large.sum(axis=1)
n_4_small = A_4_small.sum(axis=1)
P_val = []
for i, j, k, l, g, h, e, q in zip(n_1_large, n_2_large, n_3_large, n_4_large, n_1_small, n_2_small, n_3_small,
                                  n_4_small):
    table_set = np.array([[i, j, k, l], [g, h, e, q]])
    stat, pval, dof, expctd = chi2_contingency(table_set)
    P_val.append(pval)
Rej, P_value_corr = fdrcorrection(np.array(P_val), alpha=0.05, method='indep', is_sorted=False)
target_gene = np.where(P_value_corr < 0.05)[0]
pd.DataFrame(gene_names[target_gene]).to_csv("./result/gene_names_alldata", index=True)
plt.title("P_value_FDR")
plt.scatter(range(A_1.shape[0]), P_value_corr, marker=".")
plt.plot(range(A_1.shape[0]), [0.05] * A_1.shape[0], c="black")
plt.savefig("./result/P_value_FDR_alldata.jpg")
plt.show()
plt.close()
plt.title("P_value_FDR_sort_alldata")
plt.scatter(range(A_1.shape[0]), np.sort(P_value_corr), marker=".")
plt.plot(range(A_1.shape[0]), [0.05] * A_1.shape[0], c="black")
plt.savefig("./result/P_value_FDR_sort.jpg")
###### output: Figure S4 in supplement
plt.show()
plt.close()



