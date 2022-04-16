import numpy as np
import pandas as pd
from EMprogram_havingsigma import AllFuncs
import multiprocessing
import sys
from sklearn.metrics.cluster import adjusted_rand_score as ars
from multiprocessing import Pool
from functools import reduce

result_list = []


def log_result(a):
    result_list.append(a)


def initializer():
    ###### the initializer function generate the model settings among our simulation
    global n, m, In, r, u, sig, Im, Inm, N_s, delta_bin, Label, prob, C_inter
    n = 50
    m = 40
    In = np.full(n, 1)
    r = 2
    u = 70
    sig = 2
    Im = np.full(m, 1)
    N_s = 3
    delta_bin = 3.5
    Label = np.arange(N_s) + 1
    prob = np.array([1 / 3, 1 / 3, 1 / 3])
    C_inter = np.array([-2, 0, 2])
    Inm = pd.DataFrame((np.zeros(n * m) + 1).reshape((n, m)), dtype=float)
    ##### defined model settings include all parameters and sample size
    

def drnf(x, p, n):
    z = []
    ps = p.cumsum(0)
    r = np.random.uniform(0, 1, n)
    for i in np.arange(n):
        z.append(x[np.where(r[i] <= ps)[0][0]])
    return z


def Sim_meth(rep):####input is the rep-th time of repeat
    np.random.seed([50 + n + m + rep])
    print(rep)
    M_Label = np.array(drnf(Label, prob, (n * m)))
    f = lambda x: C_inter[x - 1]
    Gamma_label = pd.Series(M_Label).apply(f)
    Gamma = np.array(Gamma_label).reshape((n, m))
    E = sig * np.random.randn(n, m)
    alpha = pd.DataFrame(np.random.uniform(0, 5, n))
    alpha = alpha - np.mean(alpha)
    beta = pd.DataFrame(np.random.uniform(0, 5, m))
    beta = beta - np.mean(beta)
    Y = u * Inm + alpha.dot(Im.reshape(1, m)) + In.reshape(n, 1).dot(beta.transpose()) + Gamma + E
    mu_0 = np.mean(np.mean(Y))
    alpha_0 = pd.DataFrame(np.mean(Y, axis=1)) - np.mean(np.mean(Y))
    beta_0 = pd.DataFrame(np.mean(Y, axis=0)) - np.mean(np.mean(Y))
    C_0 = C_inter
    P_0 = prob
    sigma_0 = sig
    hat_theta = AllFuncs(Y, mu_0, alpha_0, beta_0, C_0, P_0, sigma_0).EM()
    hat_group = hat_theta["hat_label"].reshape(n * m, )
    rand_hat = ars(M_Label.tolist(), hat_group.tolist())
    C_hat = np.array(hat_theta["hat_C"])
    P_hat = np.array(hat_theta["hat_P"])
    theta_Binary = AllFuncs(Y, mu_0, alpha_0, beta_0, C_0, P_0, sigma_0).Binary_Seg(delta=delta_bin)
    group_Bin = theta_Binary["labels"].reshape(n * m, )
    rand_Bin = ars(M_Label.tolist(), group_Bin.tolist())
    if theta_Binary["N_s_est"] == N_s:
        e_bin = 1
        C_Bin = np.sort(np.array(theta_Binary["C_est"]))
        P_Bin = np.sort(np.array(theta_Binary["P_est"]))
    else:
        e_bin = 0
        C_Bin = np.zeros(N_s)
        P_Bin = np.zeros(N_s)
    theta_spec = AllFuncs(Y, mu_0, alpha_0, beta_0, C_0, P_0, sigma_0).Spec()
    group_spec = theta_spec["labels"]
    rand_spec = ars(M_Label.tolist(), group_spec.tolist())
    C_spec = np.sort(np.array(theta_spec["C_est"]))
    P_spec = np.sort(np.array(theta_spec["P_est"]))
    theta_MH = AllFuncs(Y, mu_0, alpha_0, beta_0, C_0, P_0, sigma_0).MandH()
    rand_MH = ars(M_Label.tolist(), theta_MH["group_label"].tolist())
    if np.max(theta_MH["group_label"]) == N_s:
        e_MH = 1
        C_MH = np.sort(np.array(theta_MH["C_est"]))
        P_MH = np.sort(np.array(theta_MH["P_est"]))
    else:
        e_MH = 0
        C_MH = np.zeros(N_s)
        P_MH = np.zeros(N_s)
    results = [rand_MH, C_MH, P_MH, e_MH, rand_hat, C_hat, P_hat, e_bin, rand_Bin, C_Bin, P_Bin, rand_spec, C_spec,
               P_spec]
    names = ["rand_MH", "C_MH", "P_MH", "e_MH", "rand_hat", "C_hat", "P_hat", "e_Bin", "rand_Bin", "C_Bin", "P_Bin",
             "rand_spec", "C_spec", "P_spec"]
    return dict(zip(names, results))
##### output is collection of estimation, estimation rmse and rand index of four method in Section 5.3


if __name__ == '__main__':
    core = multiprocessing.cpu_count()
    p = Pool(min(core-2, 60), initializer, ())
    N_rep = int(sys.argv[1])
    ##### For an intermediate calculation. Change above into N_rep=20 for corresponding intermediate results in results_intermediate.
    for i in range(N_rep):
        Sim_results = p.apply_async(Sim_meth, args=(i,), callback=log_result)
    p.close()
    p.join()
    #### result_list contain all estimation results of N_rep times of repeat    
    initializer()
    rand_hat = reduce(lambda a, b: a + b, list(map(lambda a: a["rand_hat"], result_list))) / N_rep
    C_hat = reduce(lambda a, b: a + b, list(map(lambda a: a["C_hat"], result_list))) / N_rep
    Chat_rmse = np.sqrt(reduce(lambda a, b: a + b, list(
        map(lambda a: (a["C_hat"] - C_inter) * (a["C_hat"] - C_inter), result_list))) / N_rep)
    P_hat = reduce(lambda a, b: a + b, list(map(lambda a: a["P_hat"], result_list))) / N_rep
    Phat_rmse = np.sqrt(reduce(lambda a, b: a + b, list(
        map(lambda a: (a["P_hat"] - prob) * (a["P_hat"] - prob), result_list))) / N_rep)
    e_Bin = reduce(lambda a, b: a + b, list(map(lambda a: a["e_Bin"], result_list)))
    rand_Bin = reduce(lambda a, b: a + b, list(map(lambda a: a["rand_Bin"], result_list))) / N_rep
    C_Bin = reduce(lambda a, b: a + b, list(map(lambda a: a["C_Bin"], result_list))) / e_Bin
    CBin_rmse = np.sqrt(reduce(lambda a, b: a + b, list(
        map(lambda a: (a["C_Bin"] - C_inter) * (a["C_Bin"] - C_inter), result_list))) / N_rep)
    P_Bin = reduce(lambda a, b: a + b, list(map(lambda a: a["P_Bin"], result_list))) / e_Bin
    PBin_rmse = np.sqrt(reduce(lambda a, b: a + b, list(
        map(lambda a: (a["P_Bin"] - prob) * (a["P_Bin"] - prob), result_list))) / N_rep)
    rand_Spec = reduce(lambda a, b: a + b, list(map(lambda a: a["rand_spec"], result_list))) / N_rep
    C_spec_result = list(map(lambda a: a["C_spec"], result_list))
    C_spec_array = np.array(C_spec_result)
    C_spec_array = C_spec_array[~np.isnan(C_spec_array).any(axis=1)]
    C_spec = np.mean(C_spec_array, axis=0)
    Cspec_rmse = np.sqrt(np.mean((C_spec_array - C_inter) ** 2, axis=0))
    P_spec_list = list(map(lambda a: a["P_spec"], result_list))
    P_spec_array = np.array(P_spec_list)
    P_spec_array = P_spec_array[~np.isnan(P_spec_array).any(axis=1)]
    P_spec = np.mean(P_spec_array, axis=0)
    Pspec_rmse = np.sqrt(np.mean((P_spec_array - prob) ** 2, axis=0))
    e_MH = reduce(lambda a, b: a + b, list(map(lambda a: a["e_MH"], result_list)))
    rand_MH = reduce(lambda a, b: a + b, list(map(lambda a: a["rand_MH"], result_list))) / N_rep
    C_MH = reduce(lambda a, b: a + b, list(map(lambda a: a["C_MH"], result_list))) / e_MH
    CMH_rmse = np.sqrt(reduce(lambda a, b: a + b, list(
        map(lambda a: (a["C_MH"] - C_inter) * (a["C_MH"] - C_inter), result_list))) / N_rep)
    P_MH = reduce(lambda a, b: a + b, list(map(lambda a: a["P_MH"], result_list))) / e_MH
    PMH_rmse = np.sqrt(reduce(lambda a, b: a + b, list(
        map(lambda a: (a["P_MH"] - prob) * (a["P_MH"] - prob), result_list))) / N_rep)
    initializer()
    true = np.append(C_inter, prob)
    EM = np.append(C_hat, P_hat)
    EM_rmse = np.append(Chat_rmse, Phat_rmse)
    Bin = np.append(C_Bin, P_Bin)
    Bin_rmse = np.append(CBin_rmse, PBin_rmse)
    Spec = np.append(C_spec, P_spec)
    Spec_rmse = np.append(Cspec_rmse, Pspec_rmse)
    MH = np.append(C_MH, P_MH)
    MH_rmse = np.append(CMH_rmse, PMH_rmse)
    ###### above is sorting each term we need from result_list
    data_methods = {
        "true value": true,
        "Our methods": EM,
        "our rmse": EM_rmse,
        "Binary": Bin,
        "Binary rmse": Bin_rmse,
        "Spectral": Spec,
        "Spectral rmse": Spec_rmse,
        "Ma and Huang": MH,
        "Ma and Huang rmse": MH_rmse
    }

    data_meds = pd.DataFrame.from_dict(data_methods, orient='index')

    data_meds.to_csv('./result/data_comparing_methods_all.csv', index=True, header=True)
    #### Table 3 in Section 5.3
    rand = dict(zip(["Our Method", "Binary", "Spectral", "Ma and Huang"], [rand_hat, rand_Bin, rand_Spec, rand_MH]))
    rand_row = pd.DataFrame.from_dict(rand, orient='index')
    rand_row.to_csv('./result/sim_methods_rand_n={}_m={}.csv'.format(n, m), index = True, header = True)
    #### the last column of Table 3 in Section 5.3
