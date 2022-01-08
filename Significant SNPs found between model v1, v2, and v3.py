from matplotlib import pyplot as plt
from pandas import read_csv
from os import listdir
from pandas import read_csv as rcsv
from sklearn.utils import shuffle

v1 = read_csv("result/significant_snps/v1.csv", index_col=0)
v2 = read_csv("result/significant_snps/v2.csv", index_col=0)
v3 = read_csv("result/significant_snps/v3.csv", index_col=0)

# v1 and v2
list(set(v1.snp.tolist()) & set(v2.snp.tolist()))
v1[v1.snp=="TBGI041358_G"]
v2[v2.snp=="TBGI041358_G"]

# v1 and v3
list(set(v1.snp.tolist()) & set(v3.snp.tolist()))

# v2 and v3
list(set(v2.snp.tolist()) & set(v3.snp.tolist()))
v2[v2.snp=="TBGI038001_C"]
v3[v3.snp=="TBGI038001_C"]

# v1, v2, and v3
list(set(v1.snp.tolist()) & set(v2.snp.tolist()) & set(v3.snp.tolist()))

# Check for Chr:Pos
genot = read_csv("result/all_snps_with_chr_and_pos.csv", index_col=0)
genot.loc[["TBGI041358_G", "TBGI038001_C"]]["chr:pos"]
genot.loc[v1.snp]["chr:pos"]
genot.loc[v2.snp]["chr:pos"]
genot.loc[v3.snp]["chr:pos"]

# Number of Significant SNPs per threshold (the justification a' = 0.025)
PATH = "./result/significant_snps/"
dir_ = [i for i in listdir(PATH) if i.startswith("v1-")]
dir_ += [i for i in listdir(PATH) if i.startswith("v2-")]
dir_ += [i for i in listdir(PATH) if i.startswith("v3-")]

c = shuffle(["tomato", "bright blue", "goldenrod", "lawn green", "bright magenta", "violet blue", "ochre"], random_state=43)
title = ["NucleoNet v1 (ABST-7)", "NucleoNet v1 (ABST-6)", "NucleoNet v1 (ABST-3)", "NucleoNet v2 (ABST-7)", "NucleoNet v2 (ABST-6)", "NucleoNet v2 (ABST-3)", "NucleoNet v3 (ABST-6)"]
dict_ = {title[i] : dir_[i] for i in range(len(dir_))}

plt.figure(figsize=(10, 7))
for x, y in enumerate(dict_.items()):
    data = rcsv(PATH + y[1], index_col=0)
    plt.plot(data.threshold, data.total_snps, '--o', color="xkcd:{}".format(c[x]), label=y[0])
plt.axvline(.025, color="grey", label="Selected threshold: .025", linewidth=1, linestyle="-.")
plt.legend()
plt.xlabel("Attention Threshold")
plt.ylabel("Number of SNPs")
# plt.savefig("result/total_snps_comparison_per_threshold.png", bbox_inches="tight", dpi=1000)