from os import listdir
from pandas import read_csv
from matplotlib import pyplot as plt

PATH = "./result/training_plot/"
data = listdir(PATH)
model = [i for i in listdir("./model/") if i.startswith("v1-")]

title = ["ABST-"+str(i) for i in range(1, 8) if i != 5]
dict_1 = {data[j] : title[i] for i, j in enumerate([0, 1, 5, 2, 4, 3])}
dict_2 = {title[i] : model[j] for i, j in enumerate([5, 0, 2, 3, 4, 1])}

best_epoch = [int(x.split("-")[2]) for x in list(dict_2.values())]
best_mse = [float(x.split("-")[4]) for x in list(dict_2.values())]

plot_fig = plt.figure(figsize=(15, 8))
for k, v in enumerate(dict_1.items()):
    x = plot_fig.add_subplot(2, 3, k+1)
    result = read_csv(PATH + v[0], index_col=0)
    mse, early_stopping = result.mse, result.early_stopping[0]
    
    plt.plot(mse[1:], label="Training MSE", c="xkcd:bright blue")
    plt.axvline(early_stopping, label="Early Stopping", color="r", linestyle='dashed')
    plt.title("${}$\n{}".format(v[1], "(Experiment 1)"))
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.ylim(2.25, 3.95)
    plt.scatter(best_epoch[k]-2, best_mse[k]-.05, c='k', marker='^', s=70,
        label="Best MSE: {:.4f} (epoch {})".format(best_mse[k], best_epoch[k]))
    plt.legend(loc="upper center")
plt.tight_layout()
plt.savefig("./result/training_plot_img/v1-training-plot.png", bbox_inches='tight', dpi=1200)