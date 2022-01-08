from os import listdir
from pandas import read_csv
from matplotlib import pyplot as plt

PATH = "./result/training_plot/"
data = listdir(PATH)
model = [i for i in listdir("./model/") if i.startswith("v2-")]
model += [i for i in listdir("./model/") if i.startswith("v3-epoch-979-mse-2.61582-attn1-xavier")]

title = ["ABST-3.2", "ABST-6.2", "ABST-7.2", "ABST-6.3"]
dict_1 = {data[j] : title[i] for i, j in enumerate([8, 7, 6, 9])}
dict_2 = {title[i] : model[j] for i, j in enumerate([1, 0, 2, 3])}

best_epoch = [int(x.split("-")[2]) for x in list(dict_2.values())]
best_mse = [float(x.split("-")[4]) for x in list(dict_2.values())]

plot_fig = plt.figure(figsize=(13, 9.5))
color = ["xkcd:wintergreen", "xkcd:wintergreen", "xkcd:wintergreen", "xkcd:golden rod"]
exp = ["(Experiment 2)", "(Experiment 2)", "(Experiment 2)", "(Experiment 3)"]
for k, v in enumerate(dict_1.items()):
    x = plot_fig.add_subplot(2, 2, k+1)
    result = read_csv(PATH + v[0], index_col=0)
    mse, early_stopping = result.mse, result.early_stopping[0]
    
    plt.plot(mse[1:], label="Training MSE", c=color[k])
    plt.axvline(early_stopping, label="Early Stopping", color="r", linestyle='dashed')
    plt.title("${}$\n{}".format(v[1].split(".")[0], exp[k]))
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.ylim(2.15, 3.65)
    plt.scatter(best_epoch[k]-2, best_mse[k]-.05, c='k', marker='^', s=70,
        label="Best MSE: {:.4f} (epoch {})".format(best_mse[k], best_epoch[k]))
    plt.legend(loc="upper center")
plt.tight_layout()
plt.savefig("./result/training_plot_img/v2-and-v3-training-plot.png", bbox_inches='tight', dpi=1000)