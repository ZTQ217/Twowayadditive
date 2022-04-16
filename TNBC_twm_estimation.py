import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from EMprogram_havingsigma import AllFuncs
from sklearn.cluster import KMeans
import seaborn as sns

Aic = []
Bic = []
Compairson = ['GEO_normal', 'GEO_tnbc']
Interaction = []
R_est = []
R_var = []
for h in Compairson:
    dataTNBC = pd.read_csv('./data/{}.csv'.format(h), index_col=0, header=0)
    #### input of dataset
    random.seed(59)
    gene_sel = random.sample(list(dataTNBC.index), 1000)
    label = np.where(np.array(gene_sel) == 'nan')[0]
    [gene_sel.remove(cElement) for cElement in [gene_sel[i] for i in label]]
    Y = pd.DataFrame(columns=dataTNBC.columns)
    for s in gene_sel:
        Y.loc[s] = np.mean(dataTNBC.loc[s], axis=0)
    n = Y.shape[0]
    m = Y.shape[1]
    Y = Y.reset_index(drop=True)
    Y.columns = np.arange(len(Y.columns))
    Inm = pd.DataFrame((np.zeros(n * m) + 1).reshape((n, m)), dtype=float)
    In = np.full(n, 1)
    Im = np.full(m, 1)
    mu_0 = np.mean(np.mean(Y))
    alpha_0 = pd.DataFrame(np.mean(Y, axis=1)) - np.mean(np.mean(Y))
    alpha_0 = alpha_0.reset_index(drop=True)
    beta_0 = pd.DataFrame(np.mean(Y, axis=0)) - np.mean(np.mean(Y))
    beta_0 = beta_0.reset_index(drop=True)
    inita = pd.DataFrame(
        Y.values - mu_0 * Inm - alpha_0.dot(Im.reshape(1, m)).values - In.reshape(n, 1).dot(beta_0.transpose()))
    sigma_0 = np.std(inita.values.reshape(n * m, ))
    nongroup_like = -np.sum(np.sum(inita * inita)) / (2 * sigma_0 ** 2) - (n * m) * np.log(sigma_0)
    AIC = [2 * (n + m) - 2 * nongroup_like]
    BIC = [2 * np.log(n * m) * (n + m) - 2 * nongroup_like]
    for N_s in np.arange(4):
        N_s = N_s + 2
        y_0 = KMeans(n_clusters=N_s).fit(inita.values.reshape(-1, 1))
        C_0 = list(y_0.cluster_centers_.reshape(1, N_s)[0])
        P_0 = list(map(lambda k: float(len(np.where(y_0.labels_ == k)[0])) / (n * m), np.arange(N_s)))
        mod = np.dot(C_0, P_0)
        C_0 = C_0 - mod
        c_0 = np.array(sorted(C_0))
        p_0 = np.array([P_0 for _, P_0 in sorted(zip(C_0, P_0))])
        estimation = AllFuncs(Y, mu_0, alpha_0, beta_0, c_0, p_0, sigma_0).EM()
        AIC.append(2 * (n + m + 2 * N_s) - 2 * estimation["Q_loss"])
        BIC.append(2 * np.log(n * m) * (n + m + 2 * N_s) - 2 * estimation["Q_loss"])
        c_em = estimation["hat_C"]
        p_em = estimation["hat_P"]
        C_em = sorted(c_em, reverse=True)
        P_em = np.array([p_em for _, p_em in sorted(zip(c_em, p_em), reverse=True)])
        alpha_em = estimation["hat_alpha"]
        alpha_em.to_csv('./result/{}_alpha_est_K={}.csv'.format(h, N_s), index=True, header=True)
        beta_em = estimation["hat_beta"]
        beta_em.to_csv('./result/{}_beta_est_K={}.csv'.format(h, N_s), index=True, header=True)
        mu_em = estimation["hat_mu"]
        sigma_em = estimation["hat_sigma"]
        r_est = [mu_em]
        r_est.extend(C_em)
        r_est.extend(P_em)
        r_est.append(sigma_em)
        names = ["hat_mu"]
        list(map(lambda h: names.extend(["hat_C{}".format(h)]), np.arange(N_s)))
        list(map(lambda h: names.extend(["hat_P{}".format(h)]), np.arange(N_s)))
        names.extend(["hat_sigma"])
        results = dict(zip(names, r_est))
        r_em = pd.DataFrame.from_dict(results, orient='index').T
        r_em.to_csv('./result/r_est_on_{}_K={}.csv'.format(h, N_s), index=True, header=True)
        group_inform = estimation["hat_label"]
        pd.DataFrame(group_inform).to_csv('./result/estimation_of_{}_label_K={}.csv'.format(h, N_s), index=True, header=True)
    print(AIC)
    Aic.append(AIC)
    print(BIC)
    Bic.append(BIC)
    integer = np.argmin(AIC) + 1
    alpha_est = pd.read_csv("./result/{}_alpha_est_K={}.csv".format(h, integer), index_col=0, header=0)
    beta_est = pd.read_csv("./result/{}_beta_est_K={}.csv".format(h, integer), index_col=0, header=0)
    r_result = pd.read_csv('./result/r_est_on_{}_K={}.csv'.format(h, integer), index_col=0, header=0)
    mu_est = r_result["hat_mu"][0]
    C_est = []
    list(map(lambda a: C_est.append(r_result["hat_C{}".format(a)][0]), np.arange(integer)))
    P_est = []
    list(map(lambda a: P_est.append(r_result["hat_P{}".format(a)][0]), np.arange(integer)))
    sigma_est = r_result["hat_sigma"][0]
    label_group = pd.read_csv('./result/estimation_of_{}_label_K={}.csv'.format(h, integer), index_col=0, header=0)
    variance = AllFuncs(Y, mu_est, alpha_est, beta_est, C_est, P_est, sigma_est).Variance()
    sd_alpha = np.sqrt(variance["var_alpha"] / m)
    sd_alpha.to_csv('./result/{}_sd_alpha_K={}.csv'.format(h, integer), index=True, header=True)
    alpha_low = alpha_est['0'] - 1.96 * sd_alpha
    alpha_upp = alpha_est['0'] + 1.96 * sd_alpha
    plt.title(r'{}_$\widehat\alpha$_line_chart'.format(h))
    plt.plot(np.arange(100), alpha_est['0'][0:100], color='black', label="hat_alpha")
    plt.fill_between(np.arange(100), alpha_low[0:100], alpha_upp[0:100], color='blue', alpha=0.25)
    plt.savefig('./result/{}_est_alpha_line_chart.jpg'.format(h))
    ### Figure S2 in supplement
    plt.show()
    plt.close()

    sd_beta = np.sqrt(variance["var_beta"] / n)
    sd_beta.to_csv('./result/{}_sd_alpha_K={}.csv'.format(h, integer), index=True, header=True)
    beta_low = beta_est['0'] - 1.96 * sd_beta
    beta_upp = beta_est['0'] + 1.96 * sd_beta
    plt.title(r'{}_$\widehat\beta$_line_chart'.format(h))
    plt.plot(np.arange(m), beta_est['0'], color='black', label="hat_beta")
    plt.fill_between(np.arange(m), beta_low, beta_upp, color='blue', alpha=0.25)
    plt.savefig('./result/{}_est_beta_line_chart.jpg'.format(h))
    #### Figure S2 in supplement
    plt.show()
    plt.close()

    hat_r = [mu_est]
    hat_r.extend(C_est)
    hat_r.extend(P_est)
    hat_r.extend([sigma_est])
    var_rr = pd.DataFrame(variance["var_pinv"])
    sd_rr = np.sqrt(np.diag(var_rr) / (n * m))
    C_deriv = np.array(P_est)[0:integer - 1] / P_est[-1]
    P_deriv = (np.array(C_est)[0:integer - 1] - C_est[-1]) / P_est[-1]
    g_deriv = np.append(C_deriv, P_deriv)
    cov = var_rr.iloc[1:(2 * integer - 1), 1:(2 * integer - 1)]
    sd_C_Ns = np.sqrt(np.dot(g_deriv, cov).dot(g_deriv) / (n * m))
    cov_P = var_rr.iloc[integer:2 * integer - 1, integer:2 * integer - 1]
    sd_P_Ns = np.sqrt(np.sum(np.sum(cov_P)) / (n * m))
    sd_rr = np.insert(sd_rr, integer, sd_C_Ns)
    sd_rr = np.insert(sd_rr, 2 * integer, sd_P_Ns)
    R_est.append(hat_r)
    R_var.append(sd_rr**2)
    r_est = []
    for k in np.arange(2 * (integer + 1)):
        r_est.append(hat_r[k])
        r_est.append(sd_rr[k])
    names = ["mu_hat", "sd_mu"]
    list(map(lambda h: names.extend(["hat_C{}".format(h+1), "sd_C{}".format(h+1)]), np.arange(integer)))
    list(map(lambda h: names.extend(["hat_P{}".format(h+1), "sd_P{}".format(h+1)]), np.arange(integer)))
    names.extend(["hat_sigma", "sd_sigma"])
    results = dict(zip(names, r_est))
    r_results = pd.DataFrame.from_dict(results, orient='index').T
    r_results.to_csv('./result/est_on_{}_K={}.csv'.format(h, integer), index=True, header=True)
    ### output: Table 6 in Section 6 when K=4
    print(r_results)
r_diff = np.abs(np.array(R_est[0]) - np.array(R_est[1]))
var_diff = R_var[0] + R_var[1]
upper_diff = r_diff + 1.96 * np.sqrt(var_diff)
lower_diff = r_diff - 1.96 * np.sqrt(var_diff)
tn_diff = np.vstack((upper_diff, lower_diff))
print(tn_diff)
names_diff = ["hat_mu_n-hat_mu_t"]
list(map(lambda h: names_diff.extend(["hat_C{}_n-hat_C{}_t".format(h, h)]), np.arange(integer)))
list(map(lambda h: names_diff.extend(["hat_P{}_n-hat_P{}_t".format(h, h)]), np.arange(integer)))
names_diff.extend(["hat_sigma_n-hat_sigma_t"])
pd.DataFrame(tn_diff, index=["upper bound", "lower bound"], columns=names_diff).to_csv('./result/est_noraml_tnbc.csv', index=True, header=True)
##### output: Table 7 in Section 6
pd.DataFrame(Aic, index=["normal", "tnbc"], columns=["K=0", "K=2", "K=3", "K=4", "K=5"]).to_csv('./result/AIC.csv', index=True, header=True)
pd.DataFrame(Bic, index=["normal", "tnbc"], columns=["K=0", "K=2", "K=3", "K=4", "K=5"]).to_csv('./result/BIC.csv', index=True, header=True)
##### output: Table 5 in Section 6
tnbc_group = pd.read_csv('./result/estimation_of_GEO_tnbc_label_K=4.csv', index_col=0, header=0)
normal_group = pd.read_csv('./result/estimation_of_GEO_normal_label_K=4.csv', index_col=0, header=0)
len(np.where(tnbc_group == 1)[0])
len(np.where(normal_group == 1)[0])


def count_1(a):
    return len(np.where(a == 1)[0])


def count_4minus1(a):
    return len(np.where(a == 4)[0]) - len(np.where(a == 1)[0])


beta_hat = pd.read_csv("./result/GEO_tnbc_beta_est_K=4.csv", index_col=0, header=0).values.reshape(198, )
large = np.where(beta_hat > 0)[0]
small = np.where(beta_hat < 0)[0]

group_tnbc = pd.read_csv('./result/estimation_of_GEO_tnbc_label_K=4.csv', index_col=0, header=0)
n_old = group_tnbc.shape[0]
m_old = group_tnbc.shape[1]
group_tnbc = pd.DataFrame((np.zeros(n_old * m_old) + 5).reshape((n_old, m_old)) - group_tnbc.values, dtype=int)
group_tnbc = group_tnbc.transpose()
n_t = m_old
m_t = n_old
sub_labels_row1 = np.where(group_tnbc == 1)[0]  # 属于第一组interaction的病人
sub_labels_col1 = np.where(group_tnbc == 1)[1]
L1_row_labels = [np.where(sub_labels_row1 == l)[0] for l in large]  # 属于beta>0的含义第一组interaction的label
S1_row_labels = [np.where(sub_labels_row1 == h)[0] for h in small]


sub_labels_row4 = np.where(group_tnbc == 4)[0]
sub_labels_col4 = np.where(group_tnbc == 4)[1]
L4_row_labels = [np.where(sub_labels_row4 == l)[0] for l in large]
S4_row_labels = [np.where(sub_labels_row4 == h)[0] for h in small]


labels_mix = np.zeros(n_t * m_t).reshape(n_t, m_t)
for s in L1_row_labels:
    for i, j in zip(sub_labels_row1[s], sub_labels_col1[s]):
        labels_mix[i][j] = 4
for s in S1_row_labels:
    for i, j in zip(sub_labels_row1[s], sub_labels_col1[s]):
        labels_mix[i][j] = 3
for s in L4_row_labels:
    for i, j in zip(sub_labels_row4[s], sub_labels_col4[s]):
        labels_mix[i][j] = 2
for s in S4_row_labels:
    for i, j in zip(sub_labels_row4[s], sub_labels_col4[s]):
        labels_mix[i][j] = 1
labels_mix = pd.DataFrame(labels_mix)

labelsmix_adjust = pd.DataFrame()
prob = []
for j in np.arange(m_t):
    num_4 = len(np.where(labels_mix[j] == 4)[0])
    num_3 = len(np.where(labels_mix[j] == 3)[0])
    num_2 = len(np.where(labels_mix[j] == 2)[0])
    num_1 = len(np.where(labels_mix[j] == 1)[0])
    renew = np.append(np.repeat(4, num_4), np.repeat(2, num_2))
    renew = np.append(renew, np.zeros(n_t - num_1 - num_2 - num_3 - num_4))
    renew = np.append(renew, np.repeat(1, num_1))
    renew = np.append(renew, np.repeat(3, num_3))
    prob.append([num_4 - num_3 + (num_1 - num_2)])
    labelsmix_adjust[j] = renew


x_lable = [""] * 765
x_lable.append("gene766")
x_lable.extend([""] * (m_t - 766))
col_num = range(0, labelsmix_adjust.shape[1])
step = 200
colbar = [col_num[i:i + step] for i in range(0, len(col_num), step)]
xlable_bar = [x_lable[i:i + step] for i in range(0, len(x_lable), step)]
colors = ["white", "pink", "faded green", "dark blue", "amber"]


for l in np.arange(len(colbar)):
    gene_label = colbar[l]
    f, ax1 = plt.subplots(figsize=(20, 6), nrows=1)
    cmap = sns.xkcd_palette(colors)
    sns.heatmap(labelsmix_adjust.iloc[:, gene_label], linewidths=0, ax=ax1, vmax=4, vmin=0, cmap=cmap)
    line = pd.DataFrame({'y': [len(large)] * len(gene_label), 'x': range(len(gene_label))})
    sns.lineplot(data=line, x="x", y="y")
    colorbar = ax1.collections[0].colorbar
    r = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks([colorbar.vmin + r / 5 * (0.5 + i) for i in range(5)])
    colorbar.set_ticklabels(["None", r"$C_4$($\hat{\beta}<0$)", r"$C_4$($\hat{\beta}>0$)", r"$C_1$($\hat{\beta}<0$)",
                             r"$C_1$($\hat{\beta}>0$)"])
    ax1.set_xticks(range(len(gene_label)))
    ax1.set_xticklabels(xlable_bar[l], fontsize=10, rotation=45)  # 设置x轴图例为空值
    ax1.set_ylabel('tumor samples')
    ax1.set_yticks([0, len(large) / 2, len(large), len(large) + (n_t - len(large)) / 2, n_t])
    ax1.set_yticklabels(["", r"$\hat{\beta}>0$", " ", r"$\hat{\beta}<0$", ""], fontsize=12, rotation=90)
    ax1.tick_params(left=False, bottom=False)
    f.savefig('./result/heatmap_file_GEO_tnbc/sns_heatmap_adjust_label_bothgroup_{}.jpg'.format(l + 1), bbox_inches='tight')
    ####### Figure S3 in supplement
