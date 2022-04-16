import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
from EMprogram_havingsigma import AllFuncs
from functools import reduce


def count(a, b, c):
    if a <= b <= c:
        cot = 1
    else:
        cot = 0
    return cot


result_list = []


def log_result(a):
    result_list.append(a)


def summery(theta_0, theta_r, var_r):
    mad_theta = np.median(np.abs(theta_r - np.mean(theta_r))) * 1.4862

    theta_true = [theta_0] * len(theta_r)
    theta_lowbound = list(map(lambda a, b: a - b * 1.96, theta_r, np.sqrt(np.array(var_r) / (n * m))))
    theta_upperbound = list(map(lambda a, b: a + b * 1.96, theta_r, np.sqrt(np.array(var_r) / (n * m))))

    cover_theta = np.sum(np.array(list(map(count, theta_lowbound, theta_true, theta_upperbound)))) / len(theta_r)

    data_theta = {'theta_0': theta_0,
                  'mean_theta_hat': np.mean(theta_r),
                  'median_theta_hat': np.median(theta_r),
                  'sd_theta_hat': np.std(theta_r),
                  'mad_theta_hat': mad_theta,
                  }
    return pd.Series(data_theta)


def drnf(x, p, n):
    z = []
    ps = p.cumsum(0)
    r = np.random.uniform(0, 1, n)
    for i in np.arange(n):
        z.append(x[np.where(r[i] <= ps)[0][0]])
    return z


def initializer(nm):
    global n, m, In, Im, Inm, u, sig, N_s, prob, C_inter, Label, r_true, alpha, beta, df
    df = 5
    n = nm[0]
    m = nm[1]
    ###### adjust value of n, m to get corresponding result
    In = np.full(n, 1)
    u = 70
    sig = 1
    Im = np.full(m, 1)
    N_s = 3
    prob = np.array([0.2, 0.3, 0.5])
    Label = np.arange(N_s) + 1
    C_inter = (np.arange(N_s - 1, dtype=float) + 1) * 2
    C_last = -np.dot(prob[np.arange(N_s - 1)], C_inter) / prob[N_s - 1]
    C_inter = np.append(C_inter, C_last)
    Inm = pd.DataFrame((np.zeros(n * m) + 1).reshape((n, m)), dtype=float)
    alpha = pd.DataFrame(np.random.uniform(0, 5, n))
    alpha = alpha - np.mean(alpha)
    beta = pd.DataFrame(np.random.uniform(0, 5, m))
    beta = beta - np.mean(beta)
    r_true = [u]
    r_true.extend(C_inter[0:(N_s - 1)])
    r_true.extend(prob[0:(N_s - 1)])
    r_true.extend([sig])


def Sim_likereal(rep):
    np.random.seed([50 + n + m + rep])
    print(rep)
    M_Label = np.array(drnf(Label, prob, (n * m)))
    f = lambda x: C_inter[x - 1]
    Gamma_label = pd.Series(M_Label).apply(f)
    Gamma = np.array(Gamma_label).reshape((n, m))
    E = np.random.standard_t(df, n*m).reshape((n, m))
    Y = u * Inm + alpha.dot(Im.reshape(1, m)) + In.reshape(n, 1).dot(beta.transpose()) + Gamma + E
    mu_0 = np.mean(np.mean(Y))
    alpha_0 = pd.DataFrame(np.mean(Y, axis=1)) - np.mean(np.mean(Y))
    beta_0 = pd.DataFrame(np.mean(Y, axis=0)) - np.mean(np.mean(Y))
    C_0 = C_inter
    P_0 = prob
    mu_initial = list(map(lambda c, p: (Y - (mu_0 + c) * Inm - alpha_0.dot(Im.reshape(1, m)) - In.reshape(n, 1).dot(beta_0.transpose())) ** 2 * p, C_0, P_0))
    sum_mu = reduce(lambda a, b: a+b, mu_initial)
    sigma_0 = np.sqrt(np.sum(np.sum(sum_mu))/(n * m))
    hat_theta = AllFuncs(Y, mu_0, alpha_0, beta_0, C_0, P_0, sigma_0).EM()
    hatalpha = hat_theta["hat_alpha"]
    hat_alpha = hat_theta["hat_alpha"].values.reshape(n, )
    bias_alpha = hat_alpha - alpha.values.reshape(n, )
    hatbeta = hat_theta["hat_beta"]
    hat_beta = hat_theta["hat_beta"].values.reshape(m, )
    bias_beta = hat_beta - beta.values.reshape(m, )
    hat_mu = hat_theta["hat_mu"]
    hat_C = hat_theta["hat_C"]
    hat_P = hat_theta["hat_P"]
    hat_sigma = hat_theta["hat_sigma"]
    em_r = [hat_theta["hat_mu"]]
    em_r.extend(hat_theta["hat_C"][0:N_s - 1])
    em_r.extend(hat_theta["hat_P"][0:N_s - 1])
    em_r.extend([hat_theta["hat_sigma"]])
    var_theo_est = AllFuncs(Y, hat_mu, hatalpha, hatbeta, hat_C, hat_P, hat_sigma).Variance()
    var_rr_est = var_theo_est["var_pinv"]
    var_alpha = var_theo_est["var_alpha"]
    var_beta = var_theo_est["var_beta"]
    resluts = [bias_alpha.tolist(), hat_alpha.tolist(), bias_beta.tolist(),
               hat_beta.tolist(), hat_mu, em_r, var_rr_est, var_alpha, var_beta]
    names = ["bias_alpha", "hat_alpha", "bias_beta", "hat_beta", "hat_mu", "em_r", "var_rr", "var_alpha", "var_beta"]
    return dict(zip(names, resluts))


if __name__ == '__main__':
    core = multiprocessing.cpu_count()
    N_rep = 1000
    NM = [[50, 40], [100, 150]]
    for nm in NM:
        p = Pool(min(60, core), initializer, (nm,))
        initializer(nm)
        for rep in range(N_rep):
            Sim_results = p.apply_async(Sim_likereal, args=(rep,), callback=log_result)
            Sim_results.get()
        p.close()
        p.join()
        Bias_alpha = list(map(lambda a: a["bias_alpha"], result_list))
        Bias_beta = list(map(lambda a: a["bias_beta"], result_list))
        alpha_hat = list(map(lambda a: a["hat_alpha"], result_list))
        beta_hat = list(map(lambda a: a["hat_beta"], result_list))
        mu_hat = np.array(list(map(lambda a: a["hat_mu"], result_list)))
        r_hat = np.array(list(map(lambda a: a["em_r"], result_list))).transpose()
        Var_rr = np.array(list(map(lambda a: np.diag(a["var_rr"]), result_list))).transpose()
        Var_alpha = list(map(lambda a: a["var_alpha"], result_list))
        Var_beta = list(map(lambda a: a["var_beta"], result_list))
        data_sum = []
        initializer(nm)
        p = Pool(min(60, core-1), initializer, (nm,))
        for i, j, k in zip(r_true, r_hat, Var_rr):
            sum_theta = p.apply_async(summery, args=(i, j, k,), callback=data_sum.append)
            sum_theta.get()
        p.close()
        p.join()
        data_est = pd.DataFrame(columns=['theta_0', 'mean_theta_hat', 'median_theta_hat', 'sd_theta_hat',
                                         'mad_theta_hat'])
        for g in data_sum:
            data_est = data_est.append(g, ignore_index=True)
        data_est.index = ["mu", "c1", "c2", "p1", "p2", "sigma"]
        print(data_est)
        data_est.to_csv("./result/data_standard_t_degree={}_n={}_m={}_K={}_realdata_settings.csv".format(df, n, m, N_s), index=True, header=True)
        ###### Table S1 in supplememt
        Sd_alpha_hat = pd.DataFrame(alpha_hat).apply(np.std, axis=0)
        SD_alpha = list(map(lambda a: np.sqrt(a / m) - Sd_alpha_hat, np.array(Var_alpha)))
        plt.title(r"$\hat\alpha$_mean_standard_t_degree={}_box_n={}_m={}_K={}".format(df, n, m, N_s))
        plt.boxplot(pd.DataFrame(Bias_alpha), showfliers=False, showmeans=True)
        plt.ylim([-0.8, 0.8])
        plt.xticks([])
        plt.savefig('./result/standard_t_only_mean_scaled_degree={}_alpha_box_n={}_m={}_K={}'.format(df, n, m, N_s))
        ###### Figure S1 in supplement
        plt.show()
        plt.close()

        Sd_beta_hat = pd.DataFrame(beta_hat-np.mean(beta_hat, axis=0)).apply(np.std, axis=0)
        SD_beta = list(map(lambda a: np.sqrt(a / n) - Sd_beta_hat, np.array(Var_beta)))

        plt.title(r"$\hat\beta$ mean box standard_t degree={} n={} m={} K={}".format(df, n, m, N_s))
        plt.boxplot(pd.DataFrame(Bias_beta), showfliers=False, showmeans=True)
        plt.ylim([-0.8, 0.8])
        plt.xticks([])
        plt.savefig('./result/standard_t_only_mean_scaled_degree={}_beta_box_n={}_m={}_K={}'.format(df, n, m, N_s))
        ##### Figure S1 in supplement
        plt.show()
        plt.close()
