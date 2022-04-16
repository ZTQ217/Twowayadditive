import numpy as np
import pandas as pd
import sys
from sklearn.metrics.cluster import adjusted_rand_score as ars
from EMprogram_havingsigma import AllFuncs
import multiprocessing
from multiprocessing import Pool
from functools import reduce


result_list = []


def log_result(a):
    result_list.append(a)


def drnf(x, p, n):
    z = []
    ps = p.cumsum(0)
    r = np.random.uniform(0, 1, n)
    for i in np.arange(n):
        z.append(x[np.where(r[i] <= ps)[0][0]])
    return z


def initializer():
    global u, sig, N_s, prob, C_inter, Label, r_true, delta_bin
    u = 70
    sig = 2
    N_s = 3
    Label = np.arange(N_s) + 1
    prob = np.array([1 / 3, 1 / 3, 1 / 3])
    C_inter = np.array([-2, 0, 2])
    r_true = [u]
    r_true.extend(C_inter[0:(N_s - 1)])
    r_true.extend(prob[0:(N_s - 1)])
    r_true.extend([sig])
    delta_bin = 3.5


def Sim_meth(rep, n, m):
    In = np.full(n, 1)
    Im = np.full(m, 1)
    Inm = pd.DataFrame((np.zeros(n * m) + 1).reshape((n, m)), dtype=float)
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
    theta_Binary = AllFuncs(Y, mu_0, alpha_0, beta_0, C_0, P_0, sigma_0).Binary_Seg(delta=delta_bin)
    group_Bin = theta_Binary["labels"].reshape(n * m, )
    rand_Bin = ars(M_Label.tolist(), group_Bin.tolist())
    theta_spec = AllFuncs(Y, mu_0, alpha_0, beta_0, C_0, P_0, sigma_0).Spec()
    group_spec = theta_spec["labels"]
    rand_spec = ars(M_Label.tolist(), group_spec.tolist())
    results = [rand_hat, rand_Bin, rand_spec]
    names = ["rand_hat", "rand_Bin", "rand_spec"]
    return dict(zip(names, results))


if __name__ == '__main__':
    N_rep = int(sys.argv[1])
    ##### For an intermediate calculation. Change above into N_rep=20 for corresponding intermediate results in results_intermediate.
    N = [50, 100, 150, 200]
    M = [40, 80, 120, 150]

    Rand = pd.DataFrame(columns=["Our methods", "Binary", "Spec"])
    for n, m in zip(N, M):
        print(n, m)
        core = multiprocessing.cpu_count()
        p = Pool(core, initializer, ())
        for rep in range(N_rep):
            ### input of Sim_meth is r-th time of repeat experiment
            Sim_results = p.apply_async(Sim_meth, args=(rep, n, m), callback=log_result)
        p.close()
        p.join()
        #### output: result list is the estimation result of three method
        initializer()
        rand_hat = reduce(lambda a, b: a + b, list(map(lambda a: a["rand_hat"], result_list))) / N_rep
        rand_Bin = reduce(lambda a, b: a + b, list(map(lambda a: a["rand_Bin"], result_list))) / N_rep
        rand_Spec = reduce(lambda a, b: a + b, list(map(lambda a: a["rand_spec"], result_list))) / N_rep
        data_rand = {"Our methods":  rand_hat,
                     "Binary":  rand_Bin,
                     "Spec": rand_Spec
                     }
        Rand = Rand.append(pd.DataFrame(data_rand, index=["n = {}, m = {}".format(n, m)]))
    print(Rand)
    Rand.to_csv("./result/sim_of_rand_goes_by_sample.csv", index=True, header=True)
    ##### Table 4 in Section 5.3

