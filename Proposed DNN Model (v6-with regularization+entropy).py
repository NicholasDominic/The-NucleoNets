import torch
from pandas import read_csv, DataFrame as df
from sklearn.metrics import r2_score
from numpy import mean, std
from statistics import variance as var
from math import sqrt
from matplotlib import pyplot as plt
from time import time
from ipynb.fs.full.neural_net import load_dataset, data_processing

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
        torch.nn.init.xavier_normal_(self.attn_fc1.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.normal_(self.attn_fc2.weight)
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
#         print("{} one hot size: {}".format(self.data_name, data.size()))

        data = self.embed(data)
#         print("{} embedding size: {}".format(self.data_name, data.size()))

        data = self.flatten(data, start_dim=1)
#         print("{} flatten size: {}".format(self.data_name, data.size()))
    
        return data

class SampleLocationWideModel(torch.nn.Module):
    def __init__(self, inp_layer, out_layer, num_embedding, embedding_dim, *args, **kwargs):
        super(SampleLocationWideModel, self).__init__()
        self.input = inp_layer
        self.output = out_layer
        self.sample_loc_fc = torch.nn.Linear(self.input, self.output)
        self.data_prepared = SampleDataPreparation("Sample Location", num_embedding, embedding_dim)
        self.init_weights()
        
    def init_weights(self, *args, **kwargs):
        torch.nn.init.normal_(self.sample_loc_fc.weight)
        torch.nn.init.normal_(self.sample_loc_fc.bias)
    
    def forward(self, sample_loc, *args, **kwargs):
        data = self.data_prepared(sample_loc)
#         print("sample_data: ", data.size())

        sample_loc_wide_model = self.sample_loc_fc(data)
#         print("sample_loc_wide_model: ", sample_loc_wide_model.size())
    
        return sample_loc_wide_model

class SampleVarietyWideModel(torch.nn.Module):
    def __init__(self, inp_layer, out_layer, num_embedding, embedding_dim, *args, **kwargs):
        super(SampleVarietyWideModel, self).__init__()
        self.input = inp_layer
        self.output = out_layer
        self.sample_var_fc = torch.nn.Linear(self.input, self.output)
        self.data_prepared = SampleDataPreparation("Sample Variety", num_embedding, embedding_dim)
        self.init_weights()
        
    def init_weights(self, *args, **kwargs):
        torch.nn.init.normal_(self.sample_var_fc.weight)
        torch.nn.init.normal_(self.sample_var_fc.bias)
    
    def forward(self, sample_loc, *args, **kwargs):
        data = self.data_prepared(sample_loc)
#         print("sample_data: ", data.size())

        sample_variety_wide_model = self.sample_var_fc(data)
#         print("sample_variety_wide_model: ", sample_variety_wide_model.size())
    
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
        self.residual = torch.nn.Identity()
        self.deep = self.mlp_hidden[1]
        self.wide = self.mlp_hidden[2] # multiply by 2 because there are sample variety + location variables
        self.attn = AttentionMechanism(self.embedding_dim[0], self.attn_hidden)
        self.concat = torch.nn.AvgPool2d((1, self.embedding_dim[0]), stride=1) # kernel_size == embedding_dim; padding==0==same
        self.fully_connected_1 = torch.nn.Linear(self.num_embedding[1], self.mlp_hidden[0])
        self.fully_connected_2 = torch.nn.Linear(self.mlp_hidden[0], self.mlp_hidden[1])
        
        # fc3, MLP from sample location and sample variety should be in the same output size as deep model
        self.fully_connected_3 = torch.nn.Linear(3 * self.embedding_dim[2], self.mlp_hidden[2])
        
        # MLP for sample location
        self.sample_location_wide_model = SampleLocationWideModel(
            3 * self.embedding_dim[2], self.mlp_hidden[2], self.num_embedding[2], self.embedding_dim[2])
        
        # MLP for sample variety
#         self.sample_variety_wide_model = SampleVarietyWideModel(
#             (max(variety)+1) * self.embedding_dim[3], self.mlp_hidden[2], self.num_embedding[3], self.embedding_dim[3])
        
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

    def forward(self, snp, snp_pos, sample_loc, *args, **kwargs):   
        snp = self.snp_embedding(snp) # * sqrt(self.embedding_dim[0])
#         print("input: ", inp_1.size())
        
        snp_pos = self.positional_embedding(snp_pos) # * sqrt(self.embedding_dim[1])
#         print("pos: ", inp_2.size())
        
        x = snp + snp_pos
#         print("input+pos: ", x.size())
        
        a = self.attn(x)
#         print("attn (softmax): ", a.size())
        
        context_vect = x*a
#         print("context vector: ", context_vect.size())
        
        context_vect += self.residual(x)
#         print("context_vect+residu: ", context_vect.size())
        
        concat = self.concat(context_vect)
#         print("concat: ", concat.size())
        
        fc1 = self.ReLU(self.fully_connected_1(concat.squeeze())) # convert from [dimA, dimB, 1] to [dimA, dimB]
#         print("fc1: ", fc1.size())
        
        fc2 = self.ReLU(self.fully_connected_2(fc1))
#         print("fc2: ", fc2.size())
    
        wide_model_1 = self.sample_location_wide_model(sample_loc)
#         wide_model_2 = self.sample_variety_wide_model(sample_variety)
        wide_deep = torch.cat((wide_model_1, fc2), dim=1)
#         print("wide_deep_concat: ", wide_deep.size())
        
        result = self.wide_deep_model(wide_deep).squeeze()
#         print("result: ", result.size())
    
        return x, a.squeeze(), result

class RegularizationAndEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(RegularizationAndEntropyLoss, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.log_softmax = torch.nn.functional.log_softmax

    def forward(self, x, reg_lambda, model_params, entropy_weight):
        l1_norm = sum(p.abs().sum() for p in model_params)
        l2_norm = sum(p.pow(2.0).sum() for p in model_params)
        regularization = (reg_lambda * l1_norm) + ((1-reg_lambda) * l2_norm)
#         print("regularization: ", regularization)
    
        entropy = self.softmax(x) * self.log_softmax(x, dim=1)
        entropy = -1.0 * entropy_weight * entropy.sum()
#         print("entropy: ", entropy)
        
        return regularization + entropy

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

max_layer_n = 6
max_lr_denom = 8
max_batchsize_n = 4
max_eval = 10

space = {
    'attn_hidden': hp.choice('attn_hidden', [i*10 for i in range(1, 4)]),
    'mlp_hidden': hp.choice('mlp_hidden', [pow(2, n+4) for n in range(1, max_layer_n)]),
    'embedding_dim' : hp.choice('embedding_dim', [32, 64]),
    'batch_size' : hp.choice('batch_size', [pow(2, n+4) for n in range(1, max_batchsize_n)]),
    'lr': hp.choice('lr', [1/x for x in [pow(10, denominator) for denominator in range(4, max_lr_denom)]]),
    'reg_lambda': hp.choice('reg_lambda', [l/10 for l in range(1, 10)])
}

space_list = {
    'attn_hidden': [i*10 for i in range(1, 4)],
    'mlp_hidden': [pow(2, n+4) for n in range(1, max_layer_n)],
    'embedding_dim' : [32, 64, 128],
    'batch_size' : [pow(2, n+4) for n in range(1, max_batchsize_n)],
    'lr':[1/x for x in [pow(10, denominator) for denominator in range(4, max_lr_denom)]],
    'reg_lambda': [l/10 for l in range(1, 10)]
}

def hyperopt(params):
    model = Model(
        [3, len_snp, 2, 2],
        [params['embedding_dim'], params['embedding_dim'], params['embedding_dim'], 0],
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
                snp, pos, loc, target = i[0][0], i[0][1], i[0][2], i[1]
                batch_start_time = time()
                optimizer.zero_grad()

                # Forward Pass
                x, attn_score, y_predict = model(snp, pos, loc)

                # Backward Pass
                loss = mse_loss(y_predict, target)
                penalty = RegularizationAndEntropyLoss()
                loss += penalty(x, params["reg_lambda"], model.parameters(), .5)
                loss.backward()
                optimizer.step()

        # EVALUATION - with validation_dataloader
        model.eval()
        for data in validation_dataloader:
            with torch.no_grad():
                x, attn, output = model(data[0][0], data[0][1], data[0][2])
                validation_mse_loss = mse_loss(output, data[1]).item()
                mse_arr.append(validation_mse_loss)
    
    score = mean(mse_arr)
    return {'loss': score, 'params': params, 'status': STATUS_OK}

params = read_csv("result/v6-best_params.csv", index_col=0)
attn_hidden = int(params["attn_hidden"].values[0])
mlp_hidden = int(params["mlp_hidden"].values[0])
embedding_dim = int(params["embedding_dim"].values[0])
batch_size = int(params["batch_size"].values[0])
lr = params["lr"].values[0]
reg_lambda = params["reg_lambda"].values[0]

# Model Initiation
torch.manual_seed(43)
model = Model(
    [3, len_snp, 2, 2],
    [embedding_dim, embedding_dim, embedding_dim, 0],
    attn_hidden, [mlp_hidden, mlp_hidden, mlp_hidden]
)

# # Model Training
# CHECK MEAN AND STDEV OF WEIGHTS
# n = dict(model.named_parameters())
# std(n["snp_embedding.weight"].tolist()[0] + \
# n["snp_embedding.weight"].tolist()[1] + \
# n["snp_embedding.weight"].tolist()[2])

def mae_loss(prediction, target):
    return torch.sum(torch.abs(prediction - target)) / len(target)

def fit(data, n_epochs, lr, batch_size, patience_num, regularization_lambda_, entropy_weight, *args, **kwargs):
    patience = patience_num
    best_model, best_train_mse = None, float("inf")
    mse_per_epoch, loss_per_epoch, mae_per_epoch, r2_score_per_epoch = [], [], [], []
    training_start_time = time()
    mse_loss = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    
    model.train()
    early_stopping = 0
    for epoch in range(1, n_epochs+1): # epoch
        mse_per_batch, loss_per_batch, r2_score_per_batch, mae_per_batch = [], [], [], []
        attention_score_all_samples = []
        epoch_start_time = time()
        print("EPOCH {}".format(epoch))
        
        for _, i in enumerate(train_dataloader): # mini batch
            snp, pos, loc, target = i[0][0], i[0][1], i[0][2], i[1]
            batch_start_time = time()
            optimizer.zero_grad()
            
            # Forward Pass
            x, attn_score, y_predict = model(snp, pos, loc)
            attention_score_all_samples.append(attn_score)

            mse_score = mse_loss(y_predict, target)
            penalty = RegularizationAndEntropyLoss()
            loss = mse_score + penalty(x, regularization_lambda_, model.parameters(), entropy_weight)

            mse_per_batch.append(mse_score.item())
            loss_per_batch.append(loss.item())

            mae_score = mae_loss(y_predict, target)
            mae_per_batch.append(mae_score.item())

            r2 = r2_score(target.tolist(), y_predict.tolist())
            r2_score_per_batch.append(r2.item())
            
            print("Batch {} MSE: {}".format(_+1, mse_score.item()))
    
            loss.backward()
            optimizer.step()
        
        print("Loss: {}".format(loss))
        print("MSE Epoch {}: {} ({:.3f} s/epoch)".format(epoch, mean(mse_per_batch), time()-epoch_start_time), end="\n\n")
        mse_per_epoch.append(mean(mse_per_batch))
        mae_per_epoch.append(mean(mae_per_batch))
        r2_score_per_epoch.append(mean(r2_score_per_batch))
        loss_per_epoch.append(mean(loss_per_batch))
#         attn_dict_per_epoch["Epoch {}".format(epoch)] = attn_list_per_batch
    
        if mean(mse_per_batch) < best_train_mse:
            best_train_mse = mean(mse_per_batch)
            best_model = model
            patience = patience_num
            torch.save(best_model.state_dict(), "model/v6-best-model-params-regularization-entropy-epoch-{}-mse-{:.5f}.pt".format(epoch, best_train_mse))
        else:
            patience -= 1
        
        if patience == 0:
            print("=" * 25)
            print("BREAK POINT")
            print("The latest MSE: {}\n".format(best_train_mse))
            print("Early stopping at epoch {}".format(epoch))
            print("=" * 25)
            early_stopping = epoch

    print("=" * 25)
    print("Total exc time: {:.3} s".format(time()-training_start_time))
    print("=" * 25)
        
    return attention_score_all_samples, best_model, mse_per_epoch, mae_per_epoch, r2_score_per_epoch, loss_per_epoch, early_stopping

a, best_model, mse, mae, r2, loss, early_stopping = fit(train_data, n_epochs=50, lr=lr, batch_size=batch_size,
                                        patience_num=10, regularization_lambda_=reg_lambda, entropy_weight=.5)

plt.figure(figsize=(8, 5))
plt.plot(loss)

plt.figure(figsize=(8, 5))
plt.plot(mse)
plt.axvline(early_stopping, label="Early Stopping", color="r", linestyle='dashed')
plt.legend()