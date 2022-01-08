from pandas import read_csv as csv, DataFrame as df, merge
from sklearn.utils import shuffle
from numpy import mean

class SaveGPTable():
    def __init__(self, snp, w_path, low_quantile, up_quantile, whisker, sample_dict, *args, **kwargs):
        super(SaveGPTable, self).__init__()
        self.all_loc = {"Kuningan" : 0, "Subang" : 1, "Citayam" : 2}
        self.significant_snp = snp
        self.save_gp_table = w_path
        self.lower_quantile = low_quantile
        self.upper_quantile = up_quantile
        self.whisker = whisker
        self.sample_dict = sample_dict
        self.load_genotype_data()
        self.load_snp_data()
        self.load_sample_data()
    
    def outlier_imputation(self, phenotype_data, *args, **kwargs):
        phenotype = phenotype_data
        q1 = phenotype.rice_yield.quantile(self.lower_quantile)
        q3 = phenotype.rice_yield.quantile(self.upper_quantile)
        iqr = q3 - q1 # Interquartile Range (IQR)
        lif = q1 - (self.whisker * iqr) # lower Inner Fence (LIF)
        lof = q1 - (self.whisker * 2 * iqr) # Lower Outer Fence (LOF)
        uif = q3 + (self.whisker * iqr) # Upper Inner Fence (UIF)
        uof = q3 + (self.whisker * 2 * iqr) # Upper Outer Fence (UOF)
        
        glob_mean = mean(phenotype.loc[(phenotype.rice_yield >= lif) & (phenotype.rice_yield <= uif)].rice_yield)
        print("Global mean (for outliers imputation): {}".format(glob_mean))
        
        mild_outlier = phenotype[((phenotype.rice_yield > uif) & (phenotype.rice_yield <= uof)) | ((phenotype.rice_yield < lif) & (phenotype.rice_yield >= lof))]
        print("Total mild outier(s): {}".format(len(mild_outlier)))
        phenotype.loc[mild_outlier.index, "rice_yield"] = glob_mean
        
        extreme_outlier = phenotype[(phenotype.rice_yield > uof) | (phenotype.rice_yield < lof)]
        print("Total extreme outier(s): {}".format(len(extreme_outlier)))
        phenotype.loc[extreme_outlier.index, "rice_yield"] = glob_mean
        return phenotype
    
    def load_genotype_data(self, *args, **kwargs):
        genotype = csv("data/RiceToolkit/app-master/data/X.csv")
        genotype.rename(columns={'sample_index':'sample_id'}, inplace=True)
        genotype.location.replace(self.all_loc, inplace=True)
        return genotype
    
    def load_phenotype_data(self, *args, **kwargs):
        phenotype = csv("data/RiceToolkit/app-master/data/Y.csv")
        phenotype.rename(columns={'sample_index':'sample_id', 'yield':'rice_yield'}, inplace=True)
        phenotype.location.replace(self.all_loc, inplace=True)
        
        # Before dropping the missing yield
#         for k,v in self.all_loc.items():
#             print("{}: {}".format(k, len(phenotype[phenotype.location==v])))
        
        phenotype = phenotype[phenotype.rice_yield!=0]
        phenotype = self.outlier_imputation(phenotype)
        return phenotype
    
    def load_snp_data(self, *args, **kwargs):
        significant_snps = self.significant_snp
        genotype = self.load_genotype_data()
        snp_data = genotype[genotype.columns[2:]]
        snp_data = snp_data[significant_snps]
        snp_dict = dict(zip([i for i in range(len(snp_data.columns))], list(snp_data.columns)))
        
        for i in list(snp_data.columns):
            snp_data.loc[:, i] = snp_data[i].apply(lambda x: round(x), snp_data[i].tolist())
            
        return snp_dict, snp_data
    
    def rename_sample(self, sample_data, *args, **kwargs):
        for inp, out in self.sample_dict.items():
            sample_data.name.replace(inp, out, inplace=True)
    
    def load_sample_data(self, *args, **kwargs):
        sample = csv("data/raw-rice-data/ind-rg-samples.csv")
        sample.drop(["Unnamed: 0", "sentrixposition", "id_source", "id_reg", "remarks"], axis=1, inplace=True)     
        self.rename_sample(sample)
        sample_idx = dict(zip(list(map(lambda x: x+1, list(sample.index))), list(sample.name)))
        return sample_idx, sample
    
    def generate(self, *args, **kwargs):
        genotype = self.load_genotype_data()
        phenotype_table = self.load_phenotype_data()
        snp_dict, snp_data = self.load_snp_data()
        sample_idx, sample_data = self.load_sample_data()
        
        genotype_table = df({
            'sample_id': list(genotype.sample_id),
            'location' : list(genotype.location),
            'snps_id' : [list(snp_dict.keys()) for i in range(len(genotype))],
            'snps': snp_data.values.tolist()
        })
        
        gp_table = shuffle(merge(genotype_table, phenotype_table, how="inner"), random_state=43)
        gp_table.rename(columns={'sample_id':'sample_name'}, inplace=True)
        gp_table.insert(0, "sample_id", gp_table.sample_name)
        gp_table.sample_name.replace(sample_idx, inplace=True)
        gp_table.to_csv("data/" + self.save_gp_table + ".csv", index=False)
        
        return genotype_table, phenotype_table, snp_data, sample_data, gp_table

sample_dict_for_renaming = {
    "37--Bio110-BC-Pir4" : "37--Bio110-BC-Pir4 (BIOSA)",
    "A1 / B1 (IR58025B)" : "IR58025 A(CMS)-B(Maintener)",
    "A2 / B2 (IR62829B)" : "IR62829 A(CMS)-B(Maintener)",
    "A3 / B3 (IR68885B)" : "IR68885 A(CMS)-B(Maintener)",
    "A4 / B4 (IR68886B)" : "IR68886 A(CMS)-B(Maintener)",
    "A5 / B5 (IR68888B)" : "IR68888 A(CMS)-B(Maintener)",
    "A6 / B6 (IR68897B)" : "IR68897 A(CMS)-B(Maintener)",
    "Ciherang-Sub1" : "Ciherang + Sub1",
    "IR 64 (kontrol indica))" : "IR 64 (kontrol indica)",
    "IR72a" : "IR35366 (IR72)",
    "Kinamaze (kontrol japonica)" : "Kinamaze ",
    "O. barthii 104384" : "O. barthii ",
    "O. glaberima 100156" : "O. glaberima ",
    "O. glaberima 10194" : "O. glaberima",
    "PK12 (S4325D-1-2-3-1)" : "S4325D-1-2-3-1",
    "PK21 (BP51-1)" : "BP51-1",
    "R14 (IR40750-82-2-2-3)" : "IR40750-82-2-2-3",
    "R2 (IR53942)" : "IR53942",
    "R3 (MTU53942)" : "MTU 9992",
    "R32 (BR158-2B-23)" : "BR 168-2B-23",
    "RH" : "Rathu Heenati (Acc. No. 11730)",
    "SWAR2" : "Swarnalata2"
}

exp1_snp = csv("result/significant_snp_exp1_full_regression.csv", index_col=0)
exp1_snp = exp1_snp.drop("location")
exp1_snp = exp1_snp[exp1_snp.pval < .05]
exp1 = SaveGPTable(exp1_snp.index.tolist(), "gp_table_significant_snps_exp1", .25, .75, 1.5, sample_dict_for_renaming)
genotype_table, phenotype_table, snp_data, sample_data, gp_table = exp1.generate()

# After dropping the missing yield, the data from Citayam is reduced by 10, leaving 413.
for k,v in {"Kuningan":0, "Subang":1, "Citayam":2}.items():
    print("{}: {}".format(k, len(phenotype_table[phenotype_table.location==v])))

exp2_snp = csv("result/significant_snp_exp2_marginal_regression.csv", index_col=0)
exp2_snp = exp2_snp[exp2_snp.pval < .05]
exp2 = SaveGPTable(exp2_snp.index.tolist(), "gp_table_significant_snps_exp2", .25, .75, 1.5, sample_dict_for_renaming)
exp2.generate()

snp_prev = ["TBGI036687_C", "TBGI050092_T", "id4009920_G", "id5014338_A", "TBGI272457_A", "id7002427_T", "id8000244_T","id10003620_T","id12006560_G"]
exp3 = SaveGPTable(snp_prev, "gp_table_significant_snps_exp3", .25, .75, 1.5, sample_dict_for_renaming)
exp3.generate()