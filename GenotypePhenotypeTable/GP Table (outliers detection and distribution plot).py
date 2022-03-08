from pandas import read_csv as csv, DataFrame as df, merge
from matplotlib.pyplot import figure, scatter, boxplot, savefig, hist, get_cmap, text, title, xlabel, ylabel, hlines, legend
from seaborn import kdeplot
from seaborn import heatmap as hm
from warnings import filterwarnings as fw
from numpy import mean
from scipy.stats import kurtosis, skew, jarque_bera

fw('ignore')
all_loc = {"Kuningan" : 0, "Subang" : 1, "Citayam" : 2}

genotype = csv("data/RiceToolkit/app-master/data/X.csv")
genotype.rename(columns={'sample_index':'sample_id'}, inplace=True)
genotype.location.replace(all_loc, inplace=True)
snp_data = genotype[genotype.columns[2:]]

for i in list(snp_data.columns):
    snp_data.loc[:, i] = snp_data[i].apply(lambda x: round(x), snp_data[i].tolist())
snp_data

figure(figsize=(10, 10))
hm(snp_data.corr(), cmap="viridis")
# savefig("result/snp_corr_heatmap.png", bbox_inches='tight', dpi=1200)

snp_dict = dict(zip([i for i in range(len(snp_data.columns))], list(snp_data.columns)))
genotype_ = df({
    'sample_id': list(genotype.sample_id),
    'location' : list(genotype.location),
    'snps_id' : [list(snp_dict.keys()) for i in range(len(genotype))],
    'snps': snp_data.values.tolist()
})

phenotype = csv("data/RiceToolkit/app-master/data/Y.csv")
phenotype.rename(columns={'sample_index':'sample_id', 'yield':'rice_yield'}, inplace=True)
phenotype.location.replace(all_loc, inplace=True)
phenotype = phenotype[phenotype.rice_yield!=0] # total zero values: 10

figure(figsize=(8, 2))
boxplot(phenotype.rice_yield, vert=False)
# savefig("boxplot.png")

q1, q3 = phenotype.rice_yield.quantile(0.25), phenotype.rice_yield.quantile(0.75)
iqr = q3 - q1 # Interquartile Range (IQR)

lif = q1 - (1.5 * iqr) # lower Inner Fence (LIF)
lof = q1 - (3 * iqr) # Lower Outer Fence (LOF)
uif = q3 + (1.5 * iqr) # Upper Inner Fence (UIF)
uof = q3 + (3 * iqr) # Upper Outer Fence (UOF)
glob_mean = mean(phenotype.loc[(phenotype.rice_yield >= lif) & (phenotype.rice_yield <= uif)].rice_yield)

mild_outlier = phenotype[((phenotype.rice_yield > uif) & (phenotype.rice_yield <= uof)) | ((phenotype.rice_yield < lif) & (phenotype.rice_yield >= lof))]
print("Total mild outlier(s): {}".format(len(mild_outlier)))
phenotype.loc[mild_outlier.index, "rice_yield"] = glob_mean

figure(figsize=(7.5, 5))
hlines(phenotype.rice_yield.describe()["mean"], -30, 730, color="k", linestyle="dashed", linewidth=3, label="mean")
hlines(q1, -30, 730, color="b", linestyle="dashed", linewidth=2, label="$Q_1$")
hlines(q3, -30, 730, color="r", linestyle="dashed", linewidth=2, label="$Q_3$")
scatter(list(phenotype.index), list(phenotype.rice_yield), c="xkcd:aquamarine")
scatter(list(mild_outlier.index), list(mild_outlier.rice_yield), c="xkcd:orange red")
legend()
title("Detected outliers in Indonesian rice yield dataset")
xlabel("Total samples")
ylabel("Yield (ton/ha)")
savefig("result/outliers.png", bbox_inches='tight', dpi=1000)

extreme_outlier = phenotype[(phenotype.rice_yield > uof) | (phenotype.rice_yield < lof)]
print("Total extreme outlier(s): {}".format(len(extreme_outlier)))
phenotype.loc[extreme_outlier.index, "rice_yield"] = glob_mean
extreme_outlier

# Yield distribution after outlier imputation
scatter(list(phenotype.index), list(phenotype.rice_yield))

sample = csv("data/raw-rice-data/ind-rg-samples.csv")
sample.drop(["Unnamed: 0", "sentrixposition", "id_source", "id_reg", "remarks"], axis=1, inplace=True)
print("Missing samples: {}".format(len(sample) - len(set(phenotype.sample_id.tolist()))))

def rename(inp, out, *args, **kwargs):
    sample.name.replace(inp, out, inplace=True)

rename("37--Bio110-BC-Pir4", "37--Bio110-BC-Pir4 (BIOSA)")
rename("A1 / B1 (IR58025B)", "IR58025 A(CMS)-B(Maintener)")
rename("A2 / B2 (IR62829B)", "IR62829 A(CMS)-B(Maintener)")
rename("A3 / B3 (IR68885B)", "IR68885 A(CMS)-B(Maintener)")
rename("A4 / B4 (IR68886B)", "IR68886 A(CMS)-B(Maintener)")
rename("A5 / B5 (IR68888B)", "IR68888 A(CMS)-B(Maintener)")
rename("A6 / B6 (IR68897B)", "IR68897 A(CMS)-B(Maintener)")
rename("Ciherang-Sub1", "Ciherang + Sub1")
rename("IR 64 (kontrol indica))", "IR 64 (kontrol indica)")
rename("IR72a", "IR35366 (IR72)")
rename("Kinamaze (kontrol japonica)", "Kinamaze ")
rename("O. barthii 104384", "O. barthii ")
rename("O. glaberima 100156", "O. glaberima ")
rename("O. glaberima 10194", "O. glaberima")
rename("PK12 (S4325D-1-2-3-1)", "S4325D-1-2-3-1")
rename("PK21 (BP51-1)", "BP51-1")
rename("R14 (IR40750-82-2-2-3)", "IR40750-82-2-2-3")
rename("R2 (IR53942)", "IR53942")
rename("R3 (MTU53942)", "MTU 9992")
rename("R32 (BR158-2B-23)", "BR 168-2B-23")
rename("RH", "Rathu Heenati (Acc. No. 11730)")
rename("SWAR2", "Swarnalata2")

sample_idx = dict(zip(list(map(lambda x: x+1, list(sample.index))), list(sample.name)))
missing_samples_key = set(sample["index"].tolist()) - set(phenotype.sample_id.tolist())
missing_samples_name = {k: sample_idx[k] for k in missing_samples_key if k in sample_idx}

gp_table = merge(genotype_, phenotype, how="inner")
gp_table.rename(columns={'sample_id':'sample_name'}, inplace=True)
gp_table.insert(0, "sample_id", gp_table.sample_name)
gp_table.sample_name.replace(sample_idx, inplace=True)
gp_table.rice_yield.describe()

# Recast GP Table to fit the Statsmodels parameter
gp_table_2 = snp_data.loc[list(gp_table.sample_id)]
gp_table_2 = gp_table_2.reset_index()
gp_table_2.drop(columns="index", inplace=True)
gp_table_2.loc[:, "location"] = gp_table.location
gp_table_2.loc[:, "variety"] = gp_table.sample_id
gp_table_2.loc[:, "rice_yield"] = gp_table.rice_yield
gp_table.rice_yield.describe()

# Advanced Data Description
# * Location
# * Total sample
# * Desc stats
# * Skewness coef.
# * Kurtosis coef.

kuningan = gp_table[gp_table.location==0]
subang = gp_table[gp_table.location==1]
citayam = gp_table[gp_table.location==2]

def plot_dist(data, save=False, save_name="", *args, **kwargs):
    figure(figsize=(8, 5))
    N, bins, patches = hist(data["rice_yield"], 20, density=True, edgecolor="white")
    jet = get_cmap('jet', len(patches))
    kdeplot(data["rice_yield"], color="k", lw=1.5)
    
    print("skewness coef.\t {}".format(skew(data["rice_yield"])))
    print("kurtosis coef.\t {}".format(kurtosis(data["rice_yield"])))
    print("jarque bera test stats.\t {}".format(jarque_bera(data["rice_yield"]).statistic))
    print("jarque bera pvalue\t {}".format(jarque_bera(data["rice_yield"]).pvalue))
    print(data["rice_yield"].describe())   
    for i in range(len(patches)):
        patches[i].set_facecolor(jet(i))     

    if save==True:
        savefig("result/rice_yield_distplot_{}.png".format(save_name), bbox_inches='tight', dpi=2000)

plot_dist(gp_table_2, True, "all")
plot_dist(kuningan, True, "kuningan")
plot_dist(subang, True, "subang")
plot_dist(citayam, True, "citayam")����
