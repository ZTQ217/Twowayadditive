Software Platform:
The code has been written using python 3.8 in IDE of pycharm professional 2020.2.5 associated with anaconda3. The file conda list.txt is the required packages of this project. 
By setting python interpreter of this project as conda python 3.8, the following python script can automatically access all packages required. (platform: Windows 10 x64).

File and Folder Specification:
main.py is the master file that obtain all tables and figures in submitted maunscript of simulation and real data in Section 5, Section 6 and supplement. 
main_intermediate.py is the master file that collate and save all intermediate results of simulation by reducing the repeat times.
The folder of result_intermediate is the collection of all intermediate version of tables and figures in simulation.
The EMprogram_havingsigma.py contains the functions of all algorithms in our submitted manuscript. The EMprogram_havingsigma.py is no needed to run along but would be called
 when other scriptis running. The simulation_1.py is mainly for results in Section 5.1 of submitted manuscript. The simulation_2.py is mainly for results in Section 5.2 of submitted manuscript. 
The simulation_3.py is mainly for results in Section 5.3 of submitted manuscript. The scripts TNBC_twm_estimation.py, real_data_twm_fdr.py and data_without_twm.py are results of 
Section 6 and additional_simulation.py is mainly for results in supplement A.1.


Numerical Results:
 All python files can run separately to get these results like following instructions:
1. Run simulation_1.py to get Figure 2, Figure 3 and Table 1 in section 5.1. 
    These can be found  named as: 
    alpha_box_n=50_m=40.png
    alpha_box_n=150_m=100.png
    alpha_box_n=100_m=200.png
    rhat_n=50_m=40_settings.csv
    rhat_n=150_m=100_settings.csv
    rhat_n=100_m=200_settings.csv
    in result folder. 
2. Run simulation_2.py to get Figure 4 and Table 2 in section 5.2. These can be found named as
    alpha_box_n=878_m=198.png
    beta_box_n=878_m=198.png
    data_real_data_setting.csv
    in result folder
3. Run simulation_3.py to get Table 3 in section 5.3. These can be found named as
    data_comparing_methods_all.csv
    sim_methods_n=50_m=40.csv in result folder. 
    Run rand_sim.py to get Table 4  in section 5.3. This can be found named as
    sim_of_rand_goes_by_sample.csv in result folder.
4. Run TNBC_twm_estimation.py to get Table 5, Table 6, Figure S2 and Figure S3 in supplement. These can be found named as:
    AIC.csv and BIC.csv
    est_on_tnbc_K=4.csv and est_on_normal_K=4.csv
    tnbc_est_alpha_line_chart.jpg 
    tnbc_est_beta_line_chart.jpg
    normal_est_alpha_line_chart.jpg 
    normal_est_beta_line_chart.jpg
    sns_heatmap_adjust_lable_bothgroup_4
    in result folder. The last one is in the folder named heatmap_file_GEO_tnbc in result.
5. Run real_data_twm_fdr.py to get Figure S4 in supplement. They can be found named as :
    P_value_FDR_sort.jpg in result folder.
6. Run data_without_twm.py to get Figure S5 in supplement. They can be found named as :
    P_value_FDR_criterion_u.jpg
    P_value_FDR_criterion_t.jpg
    P_value_bonferroni_criterion_u.jpg
    P_value_bonferroni_criterion_t.jpg
    in result folder
7. Run additional_simulation.py to get Figure S1 and Table S1 in supplement. They can be found    named as:
    data_standard_t_degree=5_n=100_m=150_K=3_realdata_settings.csv
    standard_t_only_mean_scaled_degree=5_alpha_box_n=100_m=150_K=3.png
    standard_t_only_mean_scaled_degree=5_beta_box_n=100_m=150_K=3.png
    in result folder.

Intermediate Results:
The tables and figures in folder result_intermediate are named same as the corresponding results of full simulation version. 
 For example, alpha_box_n=50_m=40.png in result_intermediate is intermediate version of  alpha_box_n=50_m=40.png in result by reducing the repeat time of experiment into 100 from 100
 among simulation_1.py.