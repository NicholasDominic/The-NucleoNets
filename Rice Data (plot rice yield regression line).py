from matplotlib import pyplot as plt
from numpy import polyfit, poly1d, linspace
from sklearn.metrics import r2_score
from ipynb.fs.full.neural_net import load_dataset, data_processing

dataset, yield_stats = load_dataset()

snp_dict = {}
snps = [str(i) for i in dataset.snps]
for i, j in enumerate(set(x for x in snps)):
    snp_dict[j] = i
len(snp_dict)

dataset.snps = dataset.snps.apply(str)
dataset.snps.replace(snp_dict, inplace=True)
dataset.snps = dataset.snps.apply(int)

x = dataset.snps
y = dataset.rice_yield
plt.figure(figsize=(10, 5))
plt.scatter(x, y, color='k')

m, b = polyfit(x, y, 1)
print(m, b)
plt.plot(x, m*x + b, color="r", linewidth=3)

x = [i for i in range(len(dataset))]
y = dataset.rice_yield
plt.figure(figsize=(10, 5))
plt.scatter(x, y, color='k')

model = poly1d(polyfit(x, y, 4))
myline = linspace(1, 700, 100)
print(model)
plt.plot(myline, model(myline), color="r", linewidth=3)

print(r2_score(y, model(x)))