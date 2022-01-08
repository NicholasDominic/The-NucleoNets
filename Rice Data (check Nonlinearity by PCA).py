from pandas import read_csv, DataFrame as df
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

data = read_csv("data/gp_table_2.csv")

figure = plt.figure(figsize=(8, 8)) # figsize(width/horizontally, height/vertically)
x = data.columns[950:954].tolist()
y = data.rice_yield

for i in range(len(x)):
    f = figure.add_subplot(2, 2, i+1) # position index always starts from 1, thus i+1
    plt.scatter(data[x[i]], y, c="xkcd:green")
    plt.title("SNPs Nonlinearity (categorical)\nSample No.{}".format(i+950))
    plt.xlabel("SNP Encodings")
    plt.ylabel("Yield (ton/ha)")
plt.tight_layout(pad=2) # margin adjusted
plt.savefig("result/nonlinearity_snp_categorical.png", dpi=1000, bbox_inches="tight")

x = data.loc[:, list(data.columns[:-1])].values
y = data.loc[:, ['rice_yield']].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=1, random_state=43)
principalComponents = pca.fit_transform(x)
pca_data = df(data=principalComponents, columns=['snp'])

pca_data["target"] = y
pca_data

plt.scatter(pca_data.snp, pca_data.target, c="xkcd:blue violet")
plt.title("SNPs Nonlinearity (continuous)")
plt.xlabel("SNP Encodings")
plt.ylabel("Yield (ton/ha)")
plt.savefig("result/nonlinearity_snp_continuous.png", dpi=1000, bbox_inches="tight")