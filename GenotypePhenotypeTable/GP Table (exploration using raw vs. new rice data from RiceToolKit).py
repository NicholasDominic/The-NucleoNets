import allel
import torch
from pandas import DataFrame as df, read_csv as csv, set_option as sopt, merge
from numpy import array, NaN
from seaborn import heatmap as hm
from matplotlib.pyplot import figure
from numpy import mean

# Set to None if you wish to see all columns and rows unfolded in Pandas DataFrame
def expand(num_cols=None, num_rows=None):
    sopt("display.max_columns", num_cols) # set max columns
    sopt('display.max_rows', num_rows) # set max rows

ind_rg = allel.read_vcf(r"C:\your_directory_here.vcf")
ind_rg_samples_id = sorted([int(i[:i.find("_")]) for i in ind_rg["samples"]])
missing_samples = list(set(ind_rg_samples_id) ^ set(list(range(1, 467+1))))
print("Sample ID: {}\nTotal Missing: {}, Available: {}/467".format(missing_samples, len(missing_samples), 467-len(missing_samples)))

count = 0
for i in ind_rg["calldata/GT"]:
    if len(i) == 451:
        count += len(i)

genotype = df(allel.GenotypeArray(ind_rg["calldata/GT"]).to_n_alt(fill=-1), index=ind_rg["variants/ID"].tolist())
genotype = genotype.replace({-1: None}).T

gt_shape = array(genotype).shape
total_snps = gt_shape[0]*gt_shape[1]

print("{} = {} (check: {})".format(gt_shape, total_snps, total_snps==count))
print("Missing rate / sample")

threshold, excluded = .2, []
for i in list(genotype.index):
    total_snp = len(genotype.columns)
    missing_snp = genotype.loc[i].isna().sum()
    rate = missing_snp/total_snp
    
    if rate > threshold:
        excluded.append(i)
    
    print("Sample no. {}: {} ({:.2f}%)".format(i, missing_snp, rate*100))
print("\nSummary | {} excluded sample(s): {}".format(len(excluded), excluded))

significant_snps = genotype[["TBGI036687", "TBGI050092", "id4009920", "id5014338", "TBGI272457", "id7002427","id8000244","id10003620","id12006560"]]
set(genotype.columns).intersection(set(significant_snps.columns))

for i in significant_snps:
    missing = genotype[i].isna().sum()
    print("{} Call Rate: {} ({:.3f}%)".format(i, missing, missing/len(genotype)*100))

genotype = genotype.dropna(axis=1) # drop all missing SNPs
genotype = genotype.astype('float') # convert from object to float data type

figure(figsize=(10, 10))
hm(genotype.corr().replace({NaN: 0}), cmap="viridis") # .figure.savefig("output.png")

snp_dict = dict(zip([i for i in range(len(genotype.columns))], list(genotype.columns)))

genotypes = df({
    'sample_id': [int(i[:i.find("_")]) for i in ind_rg["samples"]],
    'snps_id' : [list(snp_dict.keys()) for i in range(len(genotype))],
    'snps': genotype.values.tolist()
})

phenotype = csv("data/raw-rice-data/ind-rg-pheno.csv")
phenotype.rename(columns={'yield':'rice_yield'}, inplace=True)
phenotype.replace("\\N", NaN, inplace=True)

float_attr_start, float_attr_end = 5, 17 # column 5 to 17 should be in float data type
phenotype = phenotype.astype(dict(zip(
    list(phenotype.columns[float_attr_start:float_attr_end]), ["float" for i in range(float_attr_end-float_attr_start)])))
phenotype = phenotype.astype({"year":"str"})

sample = csv("data/raw-rice-data/ind-rg-samples.csv", index_col=0)
sample_idx = dict(zip(list(sample.name), list(map(lambda x: x, list(sample.index)))))
unknown_samples = [i for i in phenotype.sample_id.replace(sample_idx) if type(i) is str]
print("total unknown samples: {}".format(len(unknown_samples)))
set(unknown_samples)

total_unknown_samples = 0
un_sm = list(set(unknown_samples))
un_sm.sort()

for i in un_sm:
    total_unknown_samples += len(phenotype[phenotype.sample_id==i])
    print("Sample name: {}, [{}]".format(i, len(phenotype[phenotype.sample_id==i])))
print("Total unknown samples = {} (check = {})".format(total_unknown_samples, total_unknown_samples==len(unknown_samples)))

def rename(inp, out, *args, **kwargs):
    phenotype.sample_id.replace(inp, out, inplace=True)

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


phenotype.insert(2, "sample_name", phenotype.sample_id)
phenotype.sample_id.replace(sample_idx, inplace=True)
before_removal = phenotype.sort_values(by="sample_id")
phenotype.sort_values(by="sample_id")

total_missing_yield = 0
for i in list(sorted(set(phenotype.location))):
    missing_val = len(phenotype[phenotype.location==i].query('rice_yield.isnull()'))
    total_missing_yield += missing_val
    print("Total null values in {}: {}".format(i, missing_val))
total_missing_yield

print("Null values: {}".format(len(phenotype) - len(phenotype[phenotype.rice_yield.notnull()])))
phenotype = phenotype[phenotype.rice_yield.notna()] # drop null value in rice_yield column

print("Null values: {}".format(len(phenotype[phenotype.rice_yield==0])))
phenotype = phenotype[phenotype.rice_yield != 0] # drop yield==0

all_loc = {"Kuningan" : 0, "Subang" : 1, "Citayam" : 2}
phenotype.location.replace(all_loc, inplace=True)

q = before_removal.sort_values(by="sample_id")[["sample_id", "subspecies"]].drop_duplicates()
q.groupby("subspecies").count()

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

extreme_outlier = phenotype[(phenotype.rice_yield > uof) | (phenotype.rice_yield < lof)]
print("Total extreme outlier(s): {}".format(len(extreme_outlier)))
phenotype.loc[extreme_outlier.index, "rice_yield"] = glob_mean

gp_table = merge(phenotype, genotypes, on="sample_id", how="inner")

genotype_ = genotype
genotype_.insert(loc=0, column="sample_id", value=[int(i[:i.find("_")]) for i in ind_rg["samples"]])
phenotype_yield = phenotype[["sample_id", "rice_yield"]]
gp_table_2 = merge(phenotype_yield, genotype_, on="sample_id", how="inner")

genot = csv("data/RiceToolkit/app-master/data/X.csv")
genot.rename(columns={'sample_index':'sample_id'}, inplace=True)
genot.location.replace(all_loc, inplace=True)
figure(figsize=(10, 10))
hm(genot[genot.columns[2:]].corr(), cmap="viridis") # genot[genot.columns[2:]] means only SNPs data

ryield = csv("data/RiceToolkit/app-master/data/Y.csv")
ryield.rename(columns={'sample_index':'sample_id', 'yield':'rice_yield_x'}, inplace=True)
ryield.location.replace(all_loc, inplace=True)

m = ryield[["sample_id", "location", "rice_yield_x"]].sort_values("sample_id")
n = phenotype[["sample_id", "location", "rice_yield"]].sort_values("sample_id")
m.merge(n, how='left')

gp_table_3 = merge(phenotype, genot, on=["sample_id", "location"], how="inner")
gp_table_3[["phenotype_id", "sample_id", "sample_name", "location", "rice_yield"]]
