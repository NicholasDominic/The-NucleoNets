from pandas import read_csv, Series, DataFrame as df, read_csv
from ast import literal_eval as lev
from numpy import array, ndarray, sqrt
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
from sklearn.utils import shuffle
from warnings import filterwarnings as fw; fw("ignore")
from seaborn import heatmap as hm
from time import time
from matplotlib.pyplot import figure, xlabel, ylabel, title, savefig, tight_layout

def data_preprocessing(dataset_path, split=True, split_ratio=.85, *args, **kwargs):
    dataset = shuffle(read_csv(dataset_path), random_state=43)
    dataset["snps"] = [lev(i) for i in dataset["snps"]]
    for i in dataset.index:
        dataset.snps[i].append(int(dataset.location[i]))
        dataset.snps[i].append(int(dataset.sample_id[i]))
    X, y = dataset['snps'], dataset['rice_yield']
    
    if split:
        split = int(len(X)*split_ratio)
        X_train, X_test = X[:split], X[split:]
        X_train, X_test = array(X_train.tolist()), array(X_test.tolist())
        y_train, y_test = y[:split], y[split:]
    else:
        X_train, X_test = array(X.tolist()), 0
        y_train, y_test = y, 0
    
    return dict({
        "X_train" : X_train,
        "X_test" : X_test,
        "y_train" : y_train,
        "y_test" : y_test
    })

def elasticNet(data, a=.1, ratio=.5, intercept=True, coef=True, *args, **kwargs):
    # Fit the Elastic Net model
    reg_en = ElasticNet(alpha=a, l1_ratio=ratio, random_state=43).fit(data["X_train"], data["y_train"])
    intercept_ = reg_en.intercept_ if intercept==True else None 
    coef_ = reg_en.coef_ if coef==True else None
    
    # Prediction
    if isinstance(data["X_test"], ndarray) and isinstance(data["y_test"], Series):
        y_predict = reg_en.predict(data["X_test"])
        
        residual = [y-y_hat for (y, y_hat) in zip(data["y_test"], y_predict)]
        mse_test = mean_squared_error(data["y_test"], y_predict)
        mbe_test = sum(residual) / len(data["y_test"])
        msle_test = mean_squared_log_error(data["y_test"], [0 if i < 0 else i for i in y_predict])
        mae_test = mean_absolute_error(data["y_test"], y_predict)
        smape_test = 1 / len(data["y_test"]) * sum(list(map(lambda x, y: x/y, [abs(i) for i in residual], [(y+y_hat)/2 for (y, y_hat) in zip(data["y_test"], y_predict)])))
    else:
        y_predict, mse_test, r2_test, msle_test, residual = None, None, None, None, None
    
    return dict({
        "coef" : coef_,
        "MSE" : round(mse_test, 5),
        "RMSE" : round(sqrt(mse_test), 5),
        "MBE" : round(mbe_test, 5),
        "MAE" : round(mae_test, 5),
        "MSLE" : round(msle_test, 5),
        "SMAPE" : round(smape_test, 5)
    }), (data["y_test"].tolist(), y_predict.tolist())

score, res = elasticNet(data_preprocessing("./data/gp_table.csv"), a=.4, ratio=.05)
score, _ = elasticNet(data_preprocessing("data/gp_table_significant_snps_exp1.csv"), a=.05, ratio=.95)
score, _ = elasticNet(data_preprocessing("data/gp_table_significant_snps_exp2.csv"), a=.05, ratio=.95)
score, _ = elasticNet(data_preprocessing("data/gp_table_significant_snps_exp3.csv"), a=.05, ratio=.95)

mse, rmse, mbe, mae, msle, smape = {}, {}, {}, {}, {}, {}
start_time = time()
for l1_ratio in l1_ratio_values:
    print("- Ratio {}".format(l1_ratio))
    mse_list, rmse_list, mbe_list, mae_list, msle_list, smape_list = [], [], [], [], [], []
    for alpha in alpha_constant_values:
        result = elasticNet(data_preprocessing("./data/gp_table.csv"), a=alpha, ratio=l1_ratio)
        mse_list.append(result["MSE"])
        rmse_list.append(result["RMSE"])
        mbe_list.append(result["MBE"])
        mae_list.append(result["MAE"])
        msle_list.append(result["MSLE"])
        smape_list.append(result["SMAPE"])
        
        print("\tAlpha {} - MSE: {:.5f} | RMSE: {:.5f} | MBE: {:.5f} | MAE: {:.5f} | MSLE: {:.5f} | SMAPE: {:.5f}".format(
            alpha, result["MSE"], result["RMSE"], result["MBE"], result["MAE"], result["MSLE"], result["SMAPE"]))
    mse[l1_ratio] = mse_list
    rmse[l1_ratio] = rmse_list
    mbe[l1_ratio] = mbe_list
    mae[l1_ratio] = mae_list
    msle[l1_ratio] = msle_list
    smape[l1_ratio] = smape_list
    
print("=" * 25)
print("Total exc time: {:.3} s".format(time()-start_time))
print("=" * 25)

# Save as DataFrame
df(mse, index=alpha_constant_values).to_csv("result/elastic_net_all_samples_MSE.csv")
df(rmse, index=alpha_constant_values).to_csv("result/elastic_net_all_samples_RMSE.csv")
df(mbe, index=alpha_constant_values).to_csv("result/elastic_net_all_samples_MBE.csv")
df(mae, index=alpha_constant_values).to_csv("result/elastic_net_all_samples_MAE.csv")
df(msle, index=alpha_constant_values).to_csv("result/elastic_net_all_samples_MSLE.csv")
df(smape, index=alpha_constant_values).to_csv("result/elastic_net_all_samples_SMAPE.csv")

my_figure = figure(figsize=(10, 15)) # figsize(width/horizontally, height/vertically)
for i, j in enumerate(["MSE", "RMSE", "MBE", "MAE", "MSLE", "SMAPE"]):
    a = my_figure.add_subplot(3, 2, i+1) # position index always starts from 1, thus i+1
    hm(read_csv("result/elastic_net_all_samples_" + j + ".csv", index_col=0), cmap="viridis")
    xlabel("L1 Ratio")
    ylabel("Alpha")
    title("${}$".format(j))
tight_layout() # margin adjusted
savefig("result/elastic-net-evaluation-plot.png")
