import allel
from numpy import shape, array

# **Read VCF File**
ind_rg = allel.read_vcf(r"your_directory_here.vcf")

# **Check for Missing Rice Samples**
ind_rg_samples_id = sorted([int(i[:i.find("_")]) for i in ind_rg["samples"]])
missing_samples = list(set(ind_rg_samples_id) ^ set(list(range(1, 467+1))))
print("Sample ID: {}\nTotal Missing: {}, Available: {}/467".format(missing_samples, len(missing_samples), 467-len(missing_samples)))

# count = 0
# for i in ind_rg["calldata/GT"]:
#     if len(i) == 451:
#         count += len(i)
# count # total alleles for each SNPs


# **SNP Encoding**
# ind_rg_GT = []
# for i in ind_rg["calldata/GT"]:
#     ind_rg_GT.append([sum(j) for j in i.tolist() if -1 not in j]) # remove Indels (-1) and encode into 0/1/2

# for snps in ind_rg_GT:
#     print(all(j > 2 for j in snps)) # check if there's any encoding value more than 2
#     print(all(j < 0 for j in snps)) # check if there's any encoding value less than 0


# **Remove variants if there any missing SNPs (-1)**
# ind_rg_prec = []
# for i in ind_rg["calldata/GT"].tolist():
#     flag = 0
#     for j in i:
#         if -1 in j:
#             flag += 1
    
#     if flag == 0:
#         ind_rg_prec.append(i)

# array(ind_rg_prec).shape


# **Reshape from (variants, samples) to (samples, variants) + SNP Encoding**
# ind_rg_calldataGT = []
# for x in range(0, len(ind_rg_samples_id)):
#     ind_rg_calldataGT.append([sum(i[x]) for i in ind_rg_prec])

# array(ind_rg_calldataGT).shape

# **Read Bim/Fam/Bed File**
from pandas_plink import read_plink as rp
from pandas import read_csv as rc, DataFrame as df

bim_ind = rc('data/raw-rice-data/ind-rg.csv', usecols=[i for i in range(1, 7)])
bim_ind

# bim_ind["snps"] = ind_rg_GT
# bim_ind["alleles_total"] = [len(i) for i in ind_rg_GT]
# bim_ind.to_csv("data/preprocessed_ind_snp.csv")

# **Check Total Ref/Alt for each Chromosome**
total_alleles = 0
for i in range(1, 12+1):
    total_alleles += len(bim_ind[bim_ind.chr==i])
    print("Chromosome {}: {}".format(i, len(bim_ind[bim_ind.chr==i])))
print("TOTAL: {} (check: {})".format(total_alleles, total_alleles==len(bim_ind)))

# **Validate with 1M RGP**
(bim, fam, bed) = rp('data/3000-RGP-18M/Base.bim', verbose=False)
bim = bim.rename(columns={"chrom":"chr", "a0":"ref", "a1":"alt"})
bim[["chr", "snp", "pos", "ref", "alt"]].sample(n=10)

# intersect_between_pos = set(bim_ind["pos"]).intersection(set(bim["pos"]))
# intersect_between_pos

# intersect_between_chr = set(bim_ind["chr"]).intersection(set(bim["chr"]))
# intersect_between_chr

# bim[bim.pos.isin(intersect_between_pos)].sort_values("pos")
# bim_ind.merge(bim, on=["pos"])
# bim_ind.merge(bim, on=["chr"])
# bim_ind.merge(bim, on=["pos", "chr"])

bim_ind_pos_str = df([str(x) for x in list(bim_ind.pos)])
bim_ind_chr_str = df([str(x) for x in list(bim_ind.chr)])
bim_pos_str = df([str(x) for x in list(bim.pos)])
bim_chr_str = df([str(x) for x in list(bim.chr)])

bim_concat = bim_chr_str + ":" + bim_pos_str
bim_id_concat = bim_ind_chr_str + ":" + bim_ind_pos_str

bim["chr:pos"] = bim_concat
bim_ind["chr:pos"] = bim_id_concat

bim.head()
bim_ind.head()
bim_ind.merge(bim, on="chr:pos")
bim.merge(bim_ind, on="chr:pos")