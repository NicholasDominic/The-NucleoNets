import torch
from ast import literal_eval as lev
from sklearn.utils import shuffle
from pandas import read_csv, DataFrame as df
from sklearn.model_selection import train_test_split as tts

def load_dataset():
    gp_table = shuffle(read_csv(r"./data/gp_table.csv"), random_state=43)
    gp_table.reset_index(inplace=True, drop=True)

    # Since .csv marks Python's list as a string, we should convert it back to its original type.
    gp_table["snps"] = [lev(i) for i in gp_table["snps"]]
    gp_table["snps_id"] = [lev(i) for i in gp_table["snps_id"]]

    return gp_table, gp_table.rice_yield.describe()

def data_processing(df, test_size):
    snp, snp_pos, loc, target, variety = list(df.snps), list(df.snps_id), list(df.location), list(df.rice_yield), list(df.sample_id)
    len_snp = len(snp[0])
    max_loc = max(loc)
    max_var = max(variety)
    
    # create Tensor object
    tensor_snp = [torch.tensor(i, dtype=torch.long) for i in snp]
    tensor_pos = [torch.tensor(i, dtype=torch.long) for i in snp_pos]
    tensor_loc = [torch.tensor(i) for i in loc]
    tensor_variety = [torch.tensor(i) for i in variety]
    tensor_yield = [torch.tensor(i) for i in target]
    
    # format: [[tensor(snp), tensor(pos), tensor(loc), tensor(var)], tensor(yield)]
    dataset = [data for data in zip(zip(tensor_snp, tensor_pos, tensor_loc, tensor_variety), tensor_yield)]
    
    # Data split scheme (train : val) : test = (70 : 15) : 15
    train_data, test_data = tts(dataset, test_size=test_size, shuffle=True, random_state=43)
    print("Total train/val data: {}".format(len(train_data)))
    print("Total test data: {}".format(len(test_data)))
    
    return train_data, test_data, len_snp, max_loc, max_var