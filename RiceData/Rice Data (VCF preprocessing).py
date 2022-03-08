import allel
import pandas as pd
import numpy as np

callset  =  allel.read_vcf(r"your_directory_here.vcf", numbers={'ALT': 1})
genot = callset['calldata/GT']
gt = allel.GenotypeArray(genot)
gt.to_n_alt()
genot = pd.DataFrame(gt.to_n_alt(fill=-1))
genot.columns  = callset["samples"]
genot.index = callset["variants/ID"]
genot = genot.replace({99: None})
genot.dropna().T
