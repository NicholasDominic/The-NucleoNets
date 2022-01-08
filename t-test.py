# Null Hypothesis Significance Test (NHST)

from pandas import read_csv, merge, DataFrame
from os import listdir
from numpy import mean, std, sqrt
from scipy.stats import t, ttest_rel
from matplotlib.pyplot import boxplot, figure, subplots

def read_and_get_squared_error(data, *args, **kwargs):
    a = read_csv(PATH + data, index_col=0)
    a.loc[:, "squared_error"] = list(map(lambda x, y: (y-x)**2, a.y_true, a.y_predict))
    return a

def t_statistic(data_1, data_2, t_table_one_tailed, t_table_two_tailed, suffixes=('_1', '_2'), *args, **kwargs):
    data_1 = read_and_get_squared_error(data_1)
    data_2 = read_and_get_squared_error(data_2)
    
    df = merge(data_1, data_2, on="y_true", suffixes=suffixes)
    dof = len(df)-1
    x = df["squared_error{}".format(suffixes[0])] # OLS
    y = df["squared_error{}".format(suffixes[1])] # NucleoNet

    for alt in ["two-sided", "less", "greater"]:
        if alt == "two-sided":
            sig_level = .025
        else:
            sig_level = .05
            
        t_test, p_value = ttest_rel(y, x, alternative=alt)
        print("t-stat: {:.5f}, p-value: {:.5f} ({}, validation: {})".format(t_test, p_value, alt, p_value < sig_level))
    
#     return {
#         "number_of_observation" : number_of_observation,
#         "mean_1" : round(mean(df["squared_error{}".format(suffixes[0])]), 5),
#         "variance_1" : round(std(df["squared_error{}".format(suffixes[0])])**2, 5),
#         "mean_2" : round(mean(df["squared_error{}".format(suffixes[1])]), 5),
#         "variance_2" : round(std(df["squared_error{}".format(suffixes[1])])**2, 5),
#         "degree_of_freedom" : dof,
#         "t_test" : round(t_test, 5),
#         "p_value_one_tail" : round((t.cdf(-abs(t_test), dof)), 5),
#         "t_table_one_tail" : t_table_one_tailed,
#         "validation_one_tail" : t_test > t_table_one_tailed,
#         "p_value_two_tail" : round(2*(1 - t.cdf(abs(t_test), dof)), 5),
#         "t_table_two_tail" : t_table_two_tailed,
#         "validation_two_tail" : t_test > t_table_two_tailed
#     }

PATH = "./result/prediction/"
data = [i for i in listdir(PATH) if i.endswith("_predict.csv")]
t_table_one_tailed, t_table_two_tailed = 1.6871, 2.0262

# note: Diff. data y_true (38 data): 3.188 / data y_true (104 data): 3.447
a = read_and_get_squared_error(data[1])
b = read_and_get_squared_error(data[2])
dfx = merge(a, b, on="y_true", suffixes=('_1', '_2'))

dataset = [dfx.y_true, read_csv(PATH + data[2], index_col=0).y_true] # dataset = [38_data, 104_data]
_, ax = subplots(figsize=(8, 5))
ax.set_title('Compare')
_ = ax.boxplot(dataset, showmeans=True)

# ============= NHST for Experiment 1 =============
# NucleoNet v1 vs. OLS
t_statistic(data[1], data[2], t_table_one_tailed, t_table_two_tailed)

# NucleoNet v1 vs. OLS with Elastic Net
t_statistic(data[0], data[2], t_table_one_tailed, t_table_two_tailed)

# ============= NHST for Experiment 2 =============
# NucleoNet v2 vs. OLS
t_statistic(data[1], data[3], t_table_one_tailed, t_table_two_tailed)

# NucleoNet v2 vs. OLS with Elastic Net
t_statistic(data[0], data[3], t_table_one_tailed, t_table_two_tailed)

# ============= NHST for Experiment 3 =============
# NucleoNet v3 vs. OLS
t_statistic(data[1], data[4], t_table_one_tailed, t_table_two_tailed)

# NucleoNet v3 vs. OLS with Elastic Net
t_statistic(data[0], data[4], t_table_one_tailed, t_table_two_tailed)

