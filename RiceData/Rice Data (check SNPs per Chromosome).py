from pandas import read_csv as csv, DataFrame as df, merge
from allel import read_vcf as vcf

snp = csv("data/RiceToolkit/app-master/data/X.csv")
snp.drop(["sample_index", "location"], axis=1, inplace=True)
snp_name = list(snp.columns)
selected_rice_data = df({"snp" : snp_name})
selected_rice_data

rice_genotype = vcf(r"your_directory_here.vcf")
raw_rice_data = df({
    "chromosome" : rice_genotype["variants/CHROM"],
    "position" : rice_genotype["variants/POS"],
    "chr:pos" : list(map(
        lambda x, y: x+":"+y,
            rice_genotype["variants/CHROM"],
        [str(i) for i in rice_genotype["variants/POS"]]
    )),
    "snp" : list(map(
        lambda x, y: x+"_"+y,
            [i for i in rice_genotype["variants/ID"].tolist()],
            ['None' if i[0]=='' else i[0] for i in rice_genotype["variants/ALT"].tolist()]
    ))
})

# Renaming for uniformity
raw_rice_data.loc[raw_rice_data.snp=="8_A", "snp"] = "X8_A"
raw_rice_data.loc[raw_rice_data.snp=="9_A", "snp"] = "X9_A"

rice_genotype = merge(raw_rice_data, selected_rice_data, on='snp', how='right')
snp_prev = ["TBGI036687_C", "TBGI050092_T", "id4009920_G", "id5014338_A", "TBGI272457_A", "id7002427_T", "id8000244_T","id10003620_T","id12006560_G"]

snp = rice_genotype.set_index("snp")
snp.loc[snp_prev]

all_chr = set(rice_genotype.chromosome.astype(int))
all_chr = sorted(all_chr)

count = 0
chr_slice = []
for chr_ in all_chr:
    total_snp = len(rice_genotype[rice_genotype.chromosome==str(chr_)])
    print("Chromosome {}: {}".format(chr_, total_snp))
    count += total_snp
    chr_slice.append(count)
print("Check: {}".format(len(rice_genotype)==count))
