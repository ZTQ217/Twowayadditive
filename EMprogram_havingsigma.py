import numpy as np
import pandas as pd
import math
from functools import reduce
from sklearn.cluster import SpectralClustering


class AllFuncs(object):
    def __init__(self, Y, mu, alpha, beta, C, P, sigma):
        self.sample = Y
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.C = C
        self.P = P
        self.n = Y.shape[0]
        self.In = np.full(self.n, 1)
        self.m = Y.shape[1]
        self.Im = np.full(self.m, 1)
        self.Inm = pd.DataFrame((np.zeros(self.n * self.m) + 1).reshape((self.n, self.m)), dtype=float)
        self.sigma = sigma
        ##### define all the input of our algorithm include data and all initial value of all parameters

    def normal(self):
        mu_ijk = self.sample - (self.mu + self.C) * self.Inm - self.alpha.dot(
            self.Im.reshape(1, self.m)) - self.In.reshape(
            self.n, 1).dot(self.beta.transpose())
        return mu_ijk


    def EM(self):
        Y = self.sample
        n = self.n
        m = self.m
        In = self.In
        Im = self.Im
        mu_t = self.mu
        alpha_t = self.alpha
        beta_t = self.beta
        C_t = self.C
        P_t = self.P
        sigma_t = self.sigma
        loss_t = 10000
        diff = 1000
        while diff > 0.01:
            A_IJ = list(
                map(lambda c, p: p * np.exp(
                    -AllFuncs(Y, mu_t, alpha_t, beta_t, c, p, sigma_t).normal() ** 2 / (2 * sigma_t ** 2)), C_t, P_t))
            A_ijk = list(map(lambda a: a / sum(A_IJ), A_IJ))
            P_t = list(map(lambda a: np.sum(np.sum(a)) / (n * m), A_ijk))
            for s in range(1000):
                a_ij_c = reduce(lambda a, b: a + b, list(map(lambda a, b: a * b, A_ijk, list(C_t))))
                mu_new = np.mean(np.mean(Y)) - np.mean(np.mean(a_ij_c))
                alpha_new = pd.DataFrame(np.mean(Y, axis=1) - mu_new - np.mean(a_ij_c, axis=1))
                beta_new = pd.DataFrame(np.mean(Y, axis=0) - mu_new - np.mean(a_ij_c, axis=0))
                C_new = list(map(lambda a: np.sum(np.sum(a * (Y - np.dot(alpha_new, Im.reshape(1, m)) -
                                                              np.dot(In.reshape(n, 1), np.array(beta_new).reshape(1, m))
                                                              ))) / np.sum(np.sum(a)) - mu_new, A_ijk))
                res = np.mean((abs(np.array(C_new) - np.array(C_t))))
                C_t = C_new
                if res < 0.001:
                    break
            #    C_t.sort()
            mu_t = mu_new
            alpha_t = alpha_new
            beta_t = beta_new
            theta = np.dot(C_t, P_t)
            C_t = C_t - theta
            sum_of_part = list(map(lambda c, p: AllFuncs(Y, mu_t, alpha_t, beta_t, c, p, sigma_t).normal(), C_t, P_t))
            sigma_t = np.sqrt(reduce(lambda c, d: c + d, list(
                map(lambda a, b: np.mean(np.mean(a * (b ** 2))) * n * m / (n * m - max(n, m) + 1), A_ijk,
                    sum_of_part))))
            Q_new = list(
                map(lambda c, p: p * np.exp(
                    -AllFuncs(Y, mu_t, alpha_t, beta_t, c, p, sigma_t).normal() ** 2 / (2 * sigma_t ** 2)) / sigma_t,
                    C_t, P_t))
            q_loss = np.sum(np.sum(np.log(reduce(lambda a, b: a + b, Q_new))))
            #    print(q_loss)
            diff = abs(q_loss - loss_t)
            loss_t = q_loss

        def P_last(i, j):
            p_last = list(map(lambda a: a[j][i], A_ijk))
            return np.argmax(np.array(p_last))

        label_t = np.array([[P_last(i, j) + 1 for j in np.arange(m)] for i in np.arange(n)])
        hat_value = [mu_t, alpha_t, beta_t, C_t, P_t, sigma_t, label_t, q_loss, A_ijk]
        names = ["hat_mu", "hat_alpha", "hat_beta", "hat_C", "hat_P", "hat_sigma", "hat_label", "Q_loss", "A_ijk"]
        result = dict(zip(names, hat_value))
        return result
    #### algorithm of our model, use the input in __init__ , output is estimation of all parameters

    def Binary_Seg(self, delta):
        Z = self.sample - self.mu * self.Inm - self.alpha.dot(self.Im.reshape(1, self.m)) - self.In.reshape(self.n,
                                                                                                            1).dot(
            self.beta.transpose())
        n = self.n
        m = self.m

        Z_sort = sorted(np.array(Z).reshape(Z.shape[0] * Z.shape[1], ))

        def S(i, j, k):
            S_ik = np.var(Z_sort[i:k]) * (k - i + 1)
            if k == j:
                S_jk = 0
            else:
                S_jk = np.var(Z_sort[k:j]) * (j - k)
            S_ijk = (S_jk + S_ik) / (j - i)
            S_name = ["S_ijk", "k"]
            result = dict(zip(S_name, [S_ijk, k]))
            return result

        delta_bin = [2.0, 2.04, 2.08, 2.12, 2.16, 2.2]
        Selection = []
        Group_Bin = []
        C_Bin = []
        P_Bin = []
        K_Bin = []
        for delta in delta_bin:
            changelabel = [len(Z_sort) - 1]
            d = 0
            i = 0
            while d < len(changelabel):
                j = changelabel[d]
                if j == i: break
                if S(i, j, j)["S_ijk"] < delta:
                    i = j + 1
                    d = d + 1
                else:
                    S_ij = list(map(lambda k: S(i, j, k)["S_ijk"], np.arange(i + 1, j)))
                    S_ij = np.array(S_ij)
                    k_add = np.where(S_ij == np.min(S_ij))[0][0]
                    changelabel.extend([k_add + i])
                    changelabel = sorted(changelabel)
            N_s_est = len(changelabel)
            changepoint = [Z_sort[a] for a in changelabel]

            def G_l(x):
                grouplabs = changepoint
                grouplabs = grouplabs + [x]
                L = sorted(grouplabs)
                x_label = np.where(np.array(L) == x)[0][0]
                return x_label + 1

            Group = Z.applymap(G_l)
            Labels = np.arange(np.max(Group.values)) + 1
            C_est = [
                np.mean([Z.values[i, j] for i, j in zip(list(np.where(Group == h)[0]), list(np.where(Group == h)[1]))])
                for h in Labels]
            Res = 0
            for h, g in zip(changepoint, C_est):
                res = np.take(Z_sort, np.where(Z_sort < h)[0]) - g
                Res = Res + np.sum(res * res)
            P_est = [len(np.where(Group == h)[0]) / (Z.shape[0] * Z.shape[1]) for h in Labels]
            Selection.append(Res)
            Group_Bin.append(Group)
            C_Bin.append(C_est)
            P_Bin.append(P_est)
            K_Bin.append(N_s_est)
        res_est = np.argmin(Selection)
        Group_bin = Group_Bin[res_est]
        C_bin = C_Bin[res_est]
        P_bin = P_Bin[res_est]
        print("Bin", P_bin)
        K_bin = K_Bin[res_est]
        result = [C_bin, P_bin, Group_bin.values, K_bin]
        names = ["C_est", "P_est", "labels", "N_s_est"]
        return dict(zip(names, result))
    #### compared algorithm in Section 5.3, use the input in __init__ , output is estimation of all parameters in the method corresponding

    def Spec(self):
        Z = self.sample - self.mu * self.Inm - self.alpha.dot(self.Im.reshape(1, self.m)) - self.In.reshape(self.n,
                                                                                                            1).dot(
            self.beta.transpose())
        Z_spec = np.array(Z).reshape(-1, 1)
        N_s = len(self.P)
        clustering = SpectralClustering(n_clusters=N_s, assign_labels="discretize", random_state=0).fit(Z_spec)
        Label_spectral = pd.DataFrame(clustering.labels_.reshape(self.n, self.m))
        C_est_spectral = []
        P_est_spectral = []
        for s in np.arange(N_s):
            label_cell = np.where(Label_spectral == s)
            P_est_spectral.extend([len(label_cell[0]) / (self.n * self.m)])
            C_est_spectral.extend([np.mean(list(map(lambda i, j: Z[j][i], label_cell[0], label_cell[1])))])
        print("spec", P_est_spectral)
        names = ["C_est", "P_est", "labels"]
        estimate = [C_est_spectral, P_est_spectral, clustering.labels_]
        result = dict(zip(names, estimate))
        return result
    #### compared algorithm in Section 5.3, use the input in __init__ , output is estimation of all parameters in the method corresponding

    def MandH(self):
        Z = self.sample - self.mu * self.Inm - self.alpha.dot(self.Im.reshape(1, self.m)) - self.In.reshape(self.n,
                                                                                                            1).dot(
            self.beta.transpose())
        n = self.n
        m = self.m
        Z_MH = np.array(Z).reshape(Z.shape[0] * Z.shape[1], )
        tau = 1
        gamma = 3
        C_n = 10 * np.log(np.log(n * m))
        Lam = [1.2]
        elp = 0.001
        BIC_loglik = []
        para_sele = []
        for lam in Lam:
            def MCP(a):
                if np.abs(a) - lam > 0:
                    ST = np.sign(a) * (np.abs(a) - lam)
                else:
                    ST = 0
                if np.abs(a) <= gamma * lam:
                    return ST / (1 - 1 / (gamma * tau))
                else:
                    return a

            print(lam)
            res_MH = 10000
            e = 0
            u_t = Z_MH
            ita_t = np.zeros((n * m) ** 2).reshape(n * m, n * m)
            for j in np.arange(1, n * m):
                for i in np.arange(0, j):
                    ita_t[i, j] = u_t[i] - u_t[j]
            v_t = np.zeros((n * m) ** 2).reshape(n * m, n * m)
            while res_MH > elp:
                one_nm = np.dot(np.repeat(1, n * m).reshape(n * m, 1), np.repeat(1, n * m).reshape(1, n * m))
                represent_invers = (np.diag(np.repeat(1, n * m)) + tau * one_nm) / (n * m * tau + 1)
                ita_v = ita_t - 1 / tau * v_t
                represent_ita = list(
                    map(lambda i: np.sum(ita_v[i, (i + 1):(n * m)]) - np.sum(ita_v[0:i, i]), np.arange(n * m)))
                u_MH = np.dot(represent_invers, Z_MH + tau * np.array(represent_ita))
                I_J = np.zeros((n * m) ** 2).reshape(n * m, n * m)
                for j in np.arange(1, n * m):
                    for i in np.arange(0, j):
                        I_J[i, j] = u_MH[i] - u_MH[j]
                delta_MH = I_J + 1 / tau * v_t
                ita_MH = pd.DataFrame(delta_MH).applymap(MCP).values
                v_MH = v_t + tau * (I_J - ita_MH)
                res_MH = np.sum((I_J - ita_MH) * (I_J - ita_MH))
                e = e + 1
                u_t = u_MH
                ita_t = ita_MH
                v_t = v_MH
            ita_num = ita_t[np.triu_indices(n * m, k=1)]
            labs = np.where(ita_num == 0)[0]
            row_labs = np.triu_indices(n * m, k=1)[0][labs]
            col_labs = np.triu_indices(n * m, k=1)[1][labs]
            g = 0
            unique_row = np.unique(row_labs)
            Clusts = []
            while g < len(unique_row):
                new_clust = col_labs[np.where(row_labs == unique_row[g])]
                Clusts.append(np.append(unique_row[g], new_clust))
                # print(Clusts)
                drop_list = np.intersect1d(new_clust, unique_row)
                index = list(map(lambda a: np.where(unique_row == a)[0][0], drop_list))
                unique_row = np.delete(unique_row, index)
                g = g + 1
            bic_loglik = np.log(np.mean((Z_MH - u_t) * (Z_MH - u_t))) + C_n * np.log(n * m) / (n * m) * len(Clusts)
            BIC_loglik.append(bic_loglik)
            para_sele.append(Clusts)
        T_group = para_sele[np.argmin(BIC_loglik)]
        labels_est = np.zeros(n * m)
        C_est = []
        P_est = list(map(lambda b: len(b) / (n * m), T_group))
        for a in np.arange(len(T_group)):
            labels_est[T_group[a]] = a + 1
            C_est.extend([np.mean(Z_MH[T_group[a]])])
        Est = [labels_est, C_est, P_est]
        names = ["group_label", "C_est", "P_est"]
        return dict(zip(names, Est))
    #### compared algorithm in Section 5.3, use the input in __init__ , output is estimation of all parameters in the method corresponding

    def Variance(self):
        u = self.mu
        alpha = self.alpha
        beta = self.beta
        C_inter = self.C
        N_s = len(C_inter)
        prob = self.P
        sigma = self.sigma
        n = self.n
        m = self.m
        Y = self.sample
        mu_ijk = list(map(lambda c, p: AllFuncs(Y, u, alpha, beta, c, p, sigma).normal(), C_inter, prob))
        phi_ijk = list(map(lambda x: 1/sigma * np.exp(-(x * x) / (2*sigma**2)), mu_ijk))
        sum_phi = reduce(lambda a, b: a + b, list(map(lambda a, b: sigma ** 2 * a * b, phi_ijk, prob)))
        mu_list = list(map(lambda x, b, p: x * b * p, mu_ijk, phi_ijk, prob))
        mu = reduce(lambda a, b: a + b, mu_list) / sum_phi
        sigma_list = list(map(lambda miu, pi, p: pi * p * 1/sigma * (miu ** 2 / sigma ** 2 - 1), mu_ijk, phi_ijk, prob))
#        sigma_list = list(map(lambda miu, pi, p: pi * p * sigma * (sigma ** 2 - miu ** 2), mu_ijk, phi_ijk, prob))
        rsigma = reduce(lambda a, b: a+b, sigma_list)/sum_phi * sigma ** 2
        rc = list(map(lambda x, p: (x - mu_list[N_s - 1] / prob[N_s - 1] * p) / sum_phi, mu_list, prob))
        del rc[N_s - 1]
        rp = list(map(
            lambda x, b: ((x - phi_ijk[N_s - 1]) * (sigma ** 2) - (b - C_inter[N_s - 1]) * mu_ijk[N_s - 1] * phi_ijk[N_s - 1]) / sum_phi,
            phi_ijk, C_inter))
        del rp[N_s - 1]
        rr = [mu]
        rr.extend(rc)
        rr.extend(rp)
        rr.extend([rsigma])
        rr_vec = pd.DataFrame(list(map(lambda a: a.values.reshape(n * m, ), rr)))
        A_rr = reduce(lambda a, b: a + b, list(map(lambda x: np.dot(rr_vec[x].values.reshape(2 * N_s, 1),
                                                                    rr_vec[x].values.reshape(1, 2 * N_s)),
                                                   np.arange(n * m)))) / (n * m)

        lam_mumu = mu * mu
        bar_lam_i = np.mean(lam_mumu, axis=1)
        bar_lam_j = np.mean(lam_mumu, axis=0)
        var_alpha = 1 / np.mean(lam_mumu, axis=1)
        var_beta = 1 / np.mean(lam_mumu, axis=0)
        A_rr_inv = np.linalg.inv(A_rr)
        A_rr_pinv = np.linalg.pinv(A_rr)
        A_rr_numinv = np.diag(1/np.diag(A_rr))
        A_rr_inverse = np.linalg.pinv(A_rr + np.diag([0.001] * (2 * N_s)))
        theo_var = [var_alpha, var_beta, A_rr_inverse, A_rr_pinv, A_rr_numinv]
        name_var = ["var_alpha", "var_beta", "var_Arr", "var_pinv", "var_numinv"]

        Var = dict(zip(name_var, theo_var))
        return Var
    #### algorithm of our model, use the input in __init__ , output is estimation variance of all parameters

