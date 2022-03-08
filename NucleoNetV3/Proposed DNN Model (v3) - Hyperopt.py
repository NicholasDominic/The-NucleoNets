import torch
from pandas import read_csv, DataFrame as df
from sklearn.metrics import r2_score
from numpy import mean, std
from statistics import variance as var
from math import sqrt
from matplotlib import pyplot as plt
from time import time
from ipynb.fs.full.database import load_dataset, data_processing

torch.set_printoptions(edgeitems=4, profile="full")

# Data Loading, Transform, and Labeling
dataset, yield_stats = load_dataset()
train_data, test_data, len_snp, max_loc, max_var = data_processing(dataset, .15)

# Model Definition
class AttentionMechanism(torch.nn.Module):
    def __init__(self, embedding_dim, attn_hidden, *args, **kwargs):
        super(AttentionMechanism, self).__init__()
        self.embedding_dim = embedding_dim
        self.attn_hidden = attn_hidden
        self.ReLU = torch.nn.ReLU()
        
        self.attn_fc1 = torch.nn.Linear(self.embedding_dim, self.attn_hidden)
        self.attn_fc2 = torch.nn.Linear(self.attn_hidden, 1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.init_weights()
        
    def init_weights(self, *args, **kwargs):
        torch.nn.init.normal_(self.attn_fc1.weight)
        torch.nn.init.xavier_normal_(self.attn_fc2.weight, gain=torch.nn.init.calculate_gain("relu"))
#         torch.nn.init.normal_(self.attn_fc2.weight)
        torch.nn.init.normal_(self.attn_fc1.bias)
        torch.nn.init.normal_(self.attn_fc2.bias)
    
    def forward(self, x, *args, **kwargs):
        x = self.ReLU(self.attn_fc1(x))
        x = self.softmax(self.attn_fc2(x))
        return x

class SampleDataPreparation(torch.nn.Module):
    def __init__(self, name, num_embedding, embedding_dim, *args, **kwargs):
        super(SampleDataPreparation, self).__init__()
        self.data_name = name
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.one_hot = torch.nn.functional.one_hot
        self.embed = torch.nn.Embedding(self.num_embedding, self.embedding_dim)
        self.flatten = torch.flatten
        self.init_embedding_weights()
        
    def init_embedding_weights(self, *args, **kwargs):
        torch.nn.init.normal_(self.embed.weight)
    
    def forward(self, data, *args, **kwargs):
        if self.data_name == "Sample Variety":
            data = self.one_hot(data, num_classes=max_var+1)
        else:
            data = self.one_hot(data, num_classes=max_loc+1)
        data = self.embed(data)
        data = self.flatten(data, start_dim=1)
        return data

class SampleLocationWideModel(torch.nn.Module):
    def __init__(self, inp_layer, out_layer, num_embedding, embedding_dim, *args, **kwargs):
        super(SampleLocationWideModel, self).__init__()
        self.input = inp_layer
        self.output = out_layer
        self.sample_loc_fc1 = torch.nn.Linear(self.input, self.output)
        self.data_prepared = SampleDataPreparation("Sample Location", num_embedding, embedding_dim)
    
    def forward(self, sample_loc, *args, **kwargs):
        data = self.data_prepared(sample_loc)
        sample_loc_wide_model = self.sample_loc_fc1(data)
        return sample_loc_wide_model

class SampleVarietyWideModel(torch.nn.Module):
    def __init__(self, inp_layer, out_layer, num_embedding, embedding_dim, *args, **kwargs):
        super(SampleVarietyWideModel, self).__init__()
        self.input = inp_layer
        self.output = out_layer
        self.sample_var_fc1 = torch.nn.Linear(self.input, self.output)
        self.data_prepared = SampleDataPreparation("Sample Variety", num_embedding, embedding_dim)
    
    def forward(self, sample_loc, *args, **kwargs):
        data = self.data_prepared(sample_loc)
        sample_variety_wide_model = self.sample_var_fc1(data)
        return sample_variety_wide_model

class Model(torch.nn.Module):
    def __init__(self, num_embedding, embedding_dim, attn_hidden, mlp_hidden, *args, **kwargs):
        super(Model, self).__init__()
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.attn_hidden = attn_hidden
        self.mlp_hidden = mlp_hidden
        
        # Embedding for SNP
        self.snp_embedding = torch.nn.Embedding(self.num_embedding[0], self.embedding_dim[0])
        
        # Embedding for SNP position
        self.positional_embedding = torch.nn.Embedding(self.num_embedding[1], self.embedding_dim[1])
        
        self.one_hot = torch.nn.functional.one_hot
        self.flatten = torch.flatten
        self.deep = self.mlp_hidden[1]
        self.wide = int(self.mlp_hidden[2]/8) * 2 # multiply by 2 because there are sample variety + location variables
        self.attn = AttentionMechanism(self.embedding_dim[0], self.attn_hidden)
        self.concat = torch.nn.AvgPool2d((1, self.embedding_dim[0]), stride=1) # kernel_size == embedding_dim; padding==0==same
        self.fully_connected_1 = torch.nn.Linear(self.num_embedding[1], self.mlp_hidden[0])
        self.fully_connected_2 = torch.nn.Linear(self.mlp_hidden[0], self.mlp_hidden[1])
        
        # MLP for sample location
        self.sample_location_wide_model = SampleLocationWideModel(
            (max_loc+1) * self.embedding_dim[2], int(self.mlp_hidden[2]/8), self.num_embedding[2], self.embedding_dim[2])
        
        # MLP for sample variety
        self.sample_variety_wide_model = SampleVarietyWideModel(
            (max_var+1) * self.embedding_dim[3], int(self.mlp_hidden[2]/8), self.num_embedding[3], self.embedding_dim[3])
        
        self.wide_deep_model = torch.nn.Linear(self.wide+self.deep, 1)
        self.ReLU = torch.nn.ReLU()
        self.init_weights()

    def init_weights(self, *args, **kwargs):
        torch.nn.init.normal_(self.snp_embedding.weight)
        torch.nn.init.normal_(self.positional_embedding.weight)
        torch.nn.init.normal_(self.fully_connected_1.weight)
        torch.nn.init.normal_(self.fully_connected_2.weight)
        torch.nn.init.normal_(self.fully_connected_1.bias)
        torch.nn.init.normal_(self.fully_connected_2.bias)

    def forward(self, snp, snp_pos, sample_loc, sample_variety, *args, **kwargs):   
        snp = self.snp_embedding(snp) # * sqrt(self.embedding_dim[0])      
        snp_pos = self.positional_embedding(snp_pos) # * sqrt(self.embedding_dim[1])   
        x = snp + snp_pos
        a = self.attn(x)
        context_vect = x*a
        concat = self.concat(context_vect)
        fc1 = self.ReLU(self.fully_connected_1(concat.squeeze()))
        fc2 = self.ReLU(self.fully_connected_2(fc1))
        wide_model_1 = self.sample_location_wide_model(sample_loc)
        wide_model_2 = self.sample_variety_wide_model(sample_variety)
        wide_deep = torch.cat((wide_model_1, wide_model_2, fc2), dim=1)
        result = self.wide_deep_model(wide_deep).squeeze()
    
        return x, a.squeeze(), result

# Cross Validation and Hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from numpy.random import RandomState as rs
from numpy import array
from sklearn.model_selection import KFold

X = array([[i] for i in range(len(train_data))])
idx_dict, folds = {}, 5
for i, idx in enumerate(KFold(n_splits=folds).split(X)):
    contents = {}
    train_index, validation_index = idx
    contents["train"], contents["validation"] = list(train_index), list(validation_index)
    idx_dict[i] = contents # i = fold

print("BEFORE CROSS-VALIDATION")
print("Train data: {}".format(len(train_data)), end="\n\n")
print("CROSS VALIDATION SCHEME")
print("Train data: {}".format(len(list(map(train_data.__getitem__, idx_dict[0]["train"]))))) # [0] = first fold
print("Validation data: {}".format(len(list(map(train_data.__getitem__, idx_dict[0]["validation"]))))) # [0] = first fold

max_layer_n = 4
max_lr_denom = 7
max_batchsize_n = 3
max_eval = 10

space = {
    'attn_hidden': hp.choice('attn_hidden', [i*10 for i in range(1, 4)]),
    'mlp_hidden': hp.choice('mlp_hidden', [pow(2, n+4) for n in range(1, max_layer_n)]),
    'embedding_dim' : hp.choice('embedding_dim', [16, 32, 64, 128]),
    'batch_size' : hp.choice('batch_size', [pow(2, n+4) for n in range(0, max_batchsize_n)]),
    'lr': hp.choice('lr', [1/x for x in [pow(10, denominator) for denominator in range(4, max_lr_denom)]]),
    'lambda_1': hp.choice('lambda_1', [l/10 for l in range(1, 10)]),
    'lambda_2': hp.choice('lambda_2', [l/10 for l in range(1, 10)]),
    'regularization_weight': hp.choice('regularization_weight', [1/pow(10, l) for l in range(0, 5)]),
    'entropy_weight' : hp.choice('entropy_weight', [w/100 for w in range(1, 10)])
}

space_list = {
    'attn_hidden': [i*10 for i in range(1, 4)],
    'mlp_hidden': [pow(2, n+4) for n in range(1, max_layer_n)],
    'embedding_dim' : [16, 32, 64, 128],
    'batch_size' : [pow(2, n+4) for n in range(0, max_batchsize_n)],
    'lr':[1/x for x in [pow(10, denominator) for denominator in range(4, max_lr_denom)]],
    'lambda_1': [l/10 for l in range(1, 10)],
    'lambda_2': [l/10 for l in range(1, 10)],
    'regularization_weight': [1/pow(10, l) for l in range(0, 5)],
    'entropy_weight' : [w/100 for w in range(1, 10)]
}

class RegularizationAndEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(RegularizationAndEntropyLoss, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.log_softmax = torch.nn.functional.log_softmax

    def forward(self, x, lambda_1, lambda_2, regularization_weight, entropy_weight, model_params):
        l1_norm = sum(p.abs().sum() for p in model_params)
        l2_norm = sum(p.pow(2.0).sum() for p in model_params)
        regularization = regularization_weight * ((lambda_1 * l1_norm) + (lambda_2 * l2_norm))

        entropy = self.softmax(x) * self.log_softmax(x, dim=1)
        entropy = -1.0 * entropy_weight * entropy.sum()
        
        return regularization + entropy

def hyperopt(params):
    model = Model(
        [3, len_snp, 2, 2],
        [params['embedding_dim'], params['embedding_dim'], params['embedding_dim'], params['embedding_dim']],
        params['attn_hidden'],
        [params['mlp_hidden'], params['mlp_hidden'], params['mlp_hidden']]
    )
    
    mse_loss = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'])
    mse_arr = []
 
    for f in range(folds):
        train_data_f = list(map(train_data.__getitem__, idx_dict[f]["train"]))
        train_dataloader = torch.utils.data.DataLoader(train_data_f, batch_size=params['batch_size'], shuffle=True)
        validation_data_f = list(map(train_data.__getitem__, idx_dict[f]["validation"]))
        validation_dataloader = torch.utils.data.DataLoader(validation_data_f, batch_size=params['batch_size'], shuffle=True)
        
        # TRAINING - with train_dataloader
        model.train()   
        for epoch in range(1, 20+1): # epoch
            for _, i in enumerate(train_dataloader): # mini batch
                snp, pos, loc, var, target = i[0][0], i[0][1], i[0][2], i[0][3], i[1]
                batch_start_time = time()
                optimizer.zero_grad()

                # Forward Pass
                x, attn_score, y_predict = model(snp, pos, loc, var)

                # Backward Pass
                penalty = RegularizationAndEntropyLoss()
                loss = mse_loss(y_predict, target) +                     penalty(x, params["lambda_1"], params["lambda_2"],
                    params["regularization_weight"], params["entropy_weight"], model.parameters())
                loss.backward()
                optimizer.step()

        # EVALUATION - with validation_dataloader
        model.eval()
        for data in validation_dataloader:
            with torch.no_grad():
                _, attn, output = model(data[0][0], data[0][1], data[0][2], data[0][3])
                validation_mse_loss = mse_loss(output, data[1]).item()
                mse_arr.append(validation_mse_loss)
    
    score = mean(mse_arr)
    return {'loss': score, 'params': params, 'status': STATUS_OK}

best = fmin(
    fn=hyperopt,
    space=space,
    algo=tpe.suggest,
    max_evals=max_eval,
    trials=Trials(),
    rstate=rs(43)
)

df({k : space_list[k][best[k]] for k in space_list}, index=["value"]).to_csv("result/v3-best_params.csv")
params = read_csv("result/v3-best_params.csv", index_col=0)
