from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.api import add_constant, graphics
from pandas import read_csv as rcsv, concat, DataFrame as df
from numpy.random import randint
from numpy import array, sqrt, log as ln, var
from scipy.stats import rv_continuous as rvc
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_log_error as msle, mean_squared_error as mse, mean_absolute_error as mae
from warnings import filterwarnings as fw

fw("ignore")

data = shuffle(rcsv("data/gp_table_2.csv"), random_state=43)
data.reset_index(inplace=True, drop=True)
data.rice_yield.describe()
cv = data.rice_yield.describe()["mean"] / data.rice_yield.describe()["std"]

split = .85
train_data = data.iloc[:int(len(data)*split)]
train_data.rice_yield.describe()
train_data.rice_yield.describe()["mean"] / train_data.rice_yield.describe()["std"]

test_data = data.iloc[int(len(data)*split):]
test_data.rice_yield.describe()
test_data.rice_yield.describe()["mean"] / test_data.rice_yield.describe()["std"]

train_dependent_var, train_independent_var = train_data.rice_yield, train_data[list(data.columns[:-1])]

def get_significant_snp(model, save=False, PATH="", *args, **kwargs):
    significant_snp_pval = dict(model.pvalues[1:])
    significant_snp_coef = dict(model.params[list(significant_snp_pval.keys())])
    
    table = df({
        "pval" : list(significant_snp_pval.values()),
        "coef" : list(significant_snp_coef.values())
    }, index=[list(significant_snp_pval.keys())])
    
    if save==True:
        table.to_csv("result/significant_snp_" + PATH + ".csv")
    
    return table

def evaluation(y_true, y_predict, *args, **kwargs):
    residual = [y-y_hat for (y, y_hat) in zip(y_true, y_predict)]   
    return {
        "MSE": round(mse(y_true, y_predict), 5),
        "RMSE" : round(sqrt(mse(y_true, y_predict)), 5),
        "MBE" : round(sum(residual) / len(y_true), 5),
        "MAE" : round(mae(y_true, y_predict), 5),
        "MSLE" : round(msle(y_true, [0 if i < 0 else i for i in y_predict]), 5),
        "SMAPE" : round(100 / len(y_true) * sum(list(map(lambda x, y: x/y, [abs(i) for i in residual], [(y+y_hat)/2 for (y, y_hat) in zip(y_true, y_predict)]))), 5)
    }

# Ordinary Least Square
build_ols = OLS(train_dependent_var, add_constant(train_independent_var))
ols_model = build_ols.fit()
ols_result = ols_model.summary()

coef = df(dict(ols_model.params), index=["coef"]).T
# coef.to_csv("snps_coef_all.csv")

test_independent_var = test_data[list(test_data.columns[:-1])]
y_predict = list(ols_model.predict(add_constant(test_independent_var, has_constant='add')))
y_true = test_data.rice_yield.tolist()

x_axis = [x for x in range(len(test_data))]
plt.plot(x_axis, y_predict, label="Predict")
plt.plot(x_axis, y_true, label="True")
plt.ylim(0, 8)
plt.legend()

ols_eval = evaluation(y_true, y_predict)
df({
    "y_true" : y_true,
    "y_predict" : y_predict
}).to_csv("result/prediction/ols_predict.csv")

dependent_var, independent_var = data.rice_yield, data[list(data.columns[:-1])]
build_ols_full = OLS(dependent_var, add_constant(independent_var))
ols_model_full = build_ols_full.fit()

get_significant_snp(ols_model_full, save=True, PATH="exp1_full_regression")

# Check Significance per SNP
col = data.columns[:-1]
mr_pval, mr_coef = [], []
for i, j in enumerate(col):
    d_var = data.rice_yield
    i_var = add_constant(data[col[i]])
    build_reg = OLS(d_var, i_var)
    regression = build_reg.fit()
    mr_pval.append(regression.pvalues[1])
    mr_coef.append(regression.params[1])

# Marginal Regression
marginal_regression_table = df({
    "pval" : mr_pval,
    "coef" : mr_coef
}, index=col)

# OLS Results Function
def OLSresult(train_data, test_data, snp_cols, *args, **kwargs):   
    # Assign dependent and independent variable
    dependent_var, independent_var = train_data.rice_yield, train_data[snp_cols]
    
    # OLS
    build_ols = OLS(dependent_var, add_constant(independent_var, has_constant='add'))
    model = build_ols.fit()
    
    # Coefficient of Variation
    cov = independent_var.describe().loc["std"] / independent_var.describe().loc["mean"]
    
    # Testing
    test_independent_var = test_data[list(test_data[snp_cols])]
    y_predict = list(model.predict(add_constant(test_independent_var, has_constant='add')))
    y_true = test_data.rice_yield.tolist()
    
    return evaluation(y_true, y_predict), {"CoV" : round(cov.values.mean(), 5)}


# # Experiment 1: Significant SNPs (Full Regression)
snp_full_reg = rcsv("result/significant_snp_exp1_full_regression.csv", index_col=0)
snp_full_reg = snp_full_reg[snp_full_reg.pval < .05]
snp_full_reg = snp_full_reg.drop("location")
OLSresult(train_data, test_data, snp_full_reg.index.tolist())

# # Experiment 2: Significant SNPs (Marginal Regression)
snp_marginal_reg = rcsv("result/significant_snp_exp2_marginal_regression.csv", index_col=0)
snp_marginal_reg = snp_marginal_reg[snp_marginal_reg.pval < .05]
OLSresult(train_data, test_data, snp_marginal_reg.index.tolist())

# Check whether there are interception between SNPs in Exp. 1 and Exp. 2
set(snp_full_reg.index.tolist()) & set(snp_marginal_reg.index.tolist())

# # Experiment 3: Significant SNPs (as in previous research)
snp_prev = ["TBGI036687_C", "TBGI050092_T", "id4009920_G", "id5014338_A", "TBGI272457_A", "id7002427_T", "id8000244_T","id10003620_T","id12006560_G"]
OLSresult(train_data, test_data, snp_prev)

# # Advanced Experiment
# Exp1 + Exp2
OLSresult(train_data, test_data, snp_full_reg.index.tolist()+snp_marginal_reg.index.tolist())

# Exp2 + Exp3
OLSresult(train_data, test_data, snp_marginal_reg.index.tolist()+snp_prev)

# Exp1 + Exp3
OLSresult(train_data, test_data, snp_full_reg.index.tolist()+snp_prev)
