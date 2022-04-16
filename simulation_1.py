import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import sys
from multiprocessing import Pool
from EMprogram_havingsigma import AllFuncs


def count(a, b, c):
    if a <= b <= c:
        cot = 1
    else:
        cot = 0
    return cot


result_list = []


def log_result(a):
    result_list.append(a)


def summery(theta_0, theta_r, var_r):##### input is the true value, estimation and variance of a parameter
    mad_theta = np.median(np.abs(theta_r - np.mean(theta_r))) * 1.4862

    theta_true = [theta_0] * len(theta_r)
    theta_lowbound = list(map(lambda a, b: a - b * 1.96, theta_r, np.sqrt(np.array(var_r) / (n * m))))
    theta_upperbound = list(map(lambda a, b: a + b * 1.96, theta_r, np.sqrt(np.array(var_r) / (n * m))))

    cover_theta = np.sum(np.array(list(map(count, theta_lowbound, theta_true, theta_upperbound)))) / len(theta_r)

    data_theta = {'truth': theta_0,
                  'est': np.mean(theta_r),
                  'sd': np.std(theta_r),
                  'hat_sd': np.mean(np.sqrt(np.array(var_r) / (n * m))),
                  'cover': cover_theta
                  }
    return pd.Series(data_theta)
##### output is a row table 1


def drnf(x, p, n):
    z = []
    ps = p.cumsum(0)
    r = np.random.uniform(0, 1, n)
    for i in np.arange(n):
        z.append(x[np.where(r[i] <= ps)[0][0]])
    return z


def initializer(nm):
    ###### the initializer function generate the model settings among our simulation
    global n, m, In, Im, Inm, u, sig, N_s, prob, C_inter, Label, r_true, alpha, beta
    n = nm[0]
    m = nm[1]
    In = np.full(n, 1)
    u = 70
    sig = 2
    Im = np.full(m, 1)
    N_s = 2
    prob = np.array([0.4, 0.6])
    Label = np.arange(N_s) + 1
    C_inter = (np.arange(N_s - 1, dtype=float) + 1) * 3
    C_last = -np.dot(prob[np.arange(N_s - 1)], C_inter) / prob[N_s - 1]
    C_inter = np.append(C_inter, C_last)
    Inm = pd.DataFrame((np.zeros(n * m) + 1).reshape((n, m)), dtype=float)
    np.random.seed(50 + n + m +1224)
    alpha = pd.DataFrame(np.random.uniform(0, 2, n))
    alpha = alpha-np.mean(alpha)
    beta = pd.DataFrame(np.random.uniform(0, 2, m))
    beta = beta-np.mean(beta)
    r_true = [alpha[0][0], beta[0][0]]
    r_true.extend(C_inter[0:(N_s - 1)])
    r_true.extend(prob[0:(N_s - 1)])
    r_true.extend([u])
    r_true.extend([sig])
    ##### defined model settings include all parameters and sample size


def Sim_likereal(rep):####input is the rep-th time of repeat
    np.random.seed([50 + n + m + rep])
    print(rep)
    M_Label = np.array(drnf(Label, prob, (n * m)))
    f = lambda x: C_inter[x - 1]
    Gamma_label = pd.Series(M_Label).apply(f)
    Gamma = np.array(Gamma_label).reshape((n, m))
    E = sig * np.random.randn(n, m)
    Y = u * Inm + alpha.dot(Im.reshape(1, m)) + In.reshape(n, 1).dot(beta.transpose()) + Gamma + E
    mu_0 = np.mean(np.mean(Y))
    alpha_0 = pd.DataFrame(np.mean(Y, axis=1)) - np.mean(np.mean(Y))
    beta_0 = pd.DataFrame(np.mean(Y, axis=0)) - np.mean(np.mean(Y))
    C_0 = C_inter
    P_0 = prob
    sigma_0 = sig
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
    em_r = []
    em_r.extend(hat_theta["hat_C"][0:N_s - 1])
    em_r.extend(hat_theta["hat_P"][0:N_s - 1])
    em_r.extend([hat_theta["hat_mu"]])
    em_r.extend([hat_theta["hat_sigma"]])
    var_theo_est = AllFuncs(Y, hat_mu, hatalpha, hatbeta, hat_C, hat_P, hat_sigma).Variance()
    var_rr_est = var_theo_est["var_pinv"]
    var_alpha = var_theo_est["var_alpha"]
    var_beta = var_theo_est["var_beta"]
    resluts = [bias_alpha.tolist(), hat_alpha.tolist(), bias_beta.tolist(),
               hat_beta.tolist(), hat_mu, em_r, var_rr_est, var_alpha, var_beta]
    names = ["bias_alpha", "hat_alpha", "bias_beta", "hat_beta", "hat_mu", "em_r", "var_rr", "var_alpha", "var_beta"]
    return dict(zip(names, resluts))
#### output is result of rep-th computation include: estimation, estimation bias and variance of all the parameters in one repeat of computation.


if __name__ == '__main__':
    core = multiprocessing.cpu_count()
    N_rep = int(sys.argv[1])
    ##### For an intermediate calculation. Change above into N_rep=100 for corresponding intermediate results in results_intermediate.
    NM = [[50, 40], [150, 100], [100, 200]]
    for nm in NM:
        ##### nm is input of initializer function
        p = Pool(min(60, core), initializer, (nm,))
        initializer(nm)
        for rep in range(N_rep):
            ##### rep is the input of Sim_likereal
            Sim_results = p.apply_async(Sim_likereal, args=(rep,), callback=log_result)
            Sim_results.get()
        p.close()
        p.join()
        #### result_list contain all estimation results of N_rep times of repeat
        Bias_alpha = list(map(lambda a: a["bias_alpha"], result_list))
        Bias_beta = list(map(lambda a: a["bias_beta"], result_list))
        alpha_hat = list(map(lambda a: a["hat_alpha"], result_list))
        beta_hat = list(map(lambda a: a["hat_beta"], result_list))
        mu_hat = np.array(list(map(lambda a: a["hat_mu"], result_list)))
        r_hat = np.array(list(map(lambda a: a["em_r"], result_list))).transpose()
        Var_rr = np.array(list(map(lambda a: np.diag(a["var_rr"]), result_list))).transpose()
        Var_alpha = list(map(lambda a: a["var_alpha"], result_list))
        Var_beta = list(map(lambda a: a["var_beta"], result_list))
        ###### above is sorting each term we need from result_list

        Sd_alpha_hat = pd.DataFrame(Bias_alpha).apply(np.std, axis=0)
        SD_alpha = list(map(lambda a: np.sqrt(a / m) - Sd_alpha_hat, np.array(Var_alpha)))
        plt.subplot(211)
        plt.title(r"Box plots of $\hat\alpha-\alpha$ n={} m={}".format(n, m))
        plt.boxplot(pd.DataFrame(Bias_alpha), showfliers=False, showmeans=True)
        plt.ylim([-1.5, 1.5])
        plt.xticks([])
        plt.subplot(212)
        plt.title(r"Box plots of $\hat{}(\hat\alpha)-sd(\hat\alpha)$_sd_box_n={}_m={}".format("sd", n, m))
        plt.boxplot(pd.DataFrame(SD_alpha), showfliers=False, showmeans=True)
        plt.ylim([-0.25, 0.25])
        plt.xticks([])
        plt.savefig('./result/alpha_box_n={}_m={}'.format(n, m))
        #### Figure 2 in section 5.1
        plt.show()
        plt.close()

        Sd_beta_hat = pd.DataFrame(Bias_beta).apply(np.std, axis=0)
        SD_beta = list(map(lambda a: np.sqrt(a / n) - Sd_beta_hat, np.array(Var_beta)))

        plt.subplot(211)
        plt.title(r"Box plots of $\hat\beta-\beta$ n={} m={}".format(n, m))
        plt.boxplot(pd.DataFrame(Bias_beta), showfliers=False, showmeans=True)
        plt.ylim([-1.5, 1.5])
        plt.xticks([])
        plt.subplot(212)
        plt.title(r"Box plots of $\hat{}(\hat\beta)-sd(\hat\beta)$_sd_box_n={}_m={}".format("sd", n, m))
        plt.boxplot(pd.DataFrame(SD_beta), showfliers=False, showmeans=True)
        plt.ylim([-0.25, 0.25])
        plt.xticks([])
        plt.savefig('./result/beta_box_n={}_m={}'.format(n, m))
        ##### Figure 3 in section 5.1
        plt.show()
        plt.close()

        hat_r = np.array(Bias_alpha)[:, 0] + alpha[0][0]
        hat_r = np.vstack((hat_r, np.array(Bias_beta)[:, 0] + beta[0][0]))
        hat_r = np.vstack((hat_r, r_hat))
        permutation = [1, 2, 0, 3]
        Var_rr_new = Var_rr[permutation]
        Var_est = np.array(Var_alpha)[:, 0] * n
        Var_est = np.vstack((Var_est, np.array(Var_beta)[:, 0] * m))
        Var_est = np.vstack((Var_est, Var_rr_new))
        data_sum = []
        initializer(nm)
        p = Pool(min(60, core), initializer, (nm,))
        for i, j, k in zip(r_true, hat_r, Var_est):
            sum_theta = p.apply_async(summery, args=(i, j, k,), callback=data_sum.append)
            sum_theta.get()
        p.close()
        p.join()
        data_est = pd.DataFrame(columns=['truth', 'est', 'sd',
                                         'hat_sd', 'cover'])
        for g in data_sum:
            data_est = data_est.append(g, ignore_index=True)
        data_est.index = ["alpha_1", "beta_1", "C_1", "p_1", "mu", "sigma"]
        print(data_est)
        data_est.to_csv("./result/rhat_n={}_m={}_settings.csv".format(n, m), index=True, header=True)
        ##### Table 1 in section 5.1

