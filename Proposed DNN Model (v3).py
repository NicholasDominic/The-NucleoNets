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
        torch.nn.init.xavier_normal_(self.attn_fc1.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_normal_(self.attn_fc2.weight, gain=torch.nn.init.calculate_gain("relu"))
#         torch.nn.init.normal_(self.attn_fc1.weight)
#         torch.nn.init.normal_(self.attn_fc2.weight)
        torch.nn.init.normal_(self.attn_fc1.bias)
        torch.nn.init.normal_(self.attn_fc2.bias)
    
    def forward(self, x, *args, **kwargs):
        x = self.ReLU(self.attn_fc1(x))
#         print("attn1: ", x.size())
        x = self.softmax(self.attn_fc2(x))
#         print("attn2: ", x.size())
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
        self.residual = torch.nn.Identity()
        
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
#         print("snp data: ", snp.size())
        snp = self.snp_embedding(snp) # * sqrt(self.embedding_dim[0])  
#         print("snp data embed: ", snp.size())
        
#         print("snp pos: ", snp_pos.size())
        snp_pos = self.positional_embedding(snp_pos) # * sqrt(self.embedding_dim[1])
#         print("snp pos embed: ", snp_pos.size())
        
        x = snp + snp_pos
#         print("snp+pos: ", x.size())
        
        a = self.attn(x)
#         print("a: ", a.size)
        
        context_vect = x*a
#         print("context_vect: ", context_vect.size())

#         context_vect += self.residual(x)
#         print("context_vect+residu: ", context_vect.size())
        
        concat = self.concat(context_vect)
#         print("concat: ", concat.size())
        
        fc1 = self.ReLU(self.fully_connected_1(concat.squeeze())) # convert from [dimA, dimB, 1] to [dimA, dimB]
#         print("fc1: ", fc1.size())
        
        fc2 = self.ReLU(self.fully_connected_2(fc1))
#         print("fc2: ", fc2.size())
        
#         print("sample loc: ", sample_loc.size())
        wide_model_1 = self.sample_location_wide_model(sample_loc)
#         print("wide_model_1: ", wide_model_1.size())
        
#         print("sample variety: ", sample_variety.size())
        wide_model_2 = self.sample_variety_wide_model(sample_variety)
#         print("wide_model_2: ", wide_model_2.size())
        
        wide_deep = torch.cat((wide_model_1, wide_model_2, fc2), dim=1)
#         print("wide_deep_concat: ", wide_deep.size())
#         print("wide deep:\n ", wide_deep)
        
        result = self.wide_deep_model(wide_deep).squeeze()
#         print("result:\n ", result)
#         result = self.wide_deep_model2(result)
#         result = self.wide_deep_model3(result)
#         result = self.wide_deep_model4(result).squeeze()
#         print("result: ", result.size())
    
        return x, a.squeeze(), result

class RegularizationAndEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(RegularizationAndEntropyLoss, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.log_softmax = torch.nn.functional.log_softmax

    def forward(self, x, lambda_1, lambda_2, w, entropy_weight, model_params):
        l1_norm = sum(p.abs().sum() for p in model_params)
        l2_norm = sum(p.pow(2.0).sum() for p in model_params)
        regularization = w * ((lambda_1 * l1_norm) + (lambda_2 * l2_norm))

        entropy = self.softmax(x) * self.log_softmax(x, dim=1)
        entropy = -1.0 * entropy_weight * entropy.sum()
        
        return regularization + entropy

params = read_csv("result/v3-best_params.csv", index_col=0)
attn_hidden = int(params["attn_hidden"].values[0])
mlp_hidden = int(params["mlp_hidden"].values[0])
embedding_dim = int(params["embedding_dim"].values[0])
batch_size = int(params["batch_size"].values[0])
lr = params["lr"].values[0]
lambda_1 = params["lambda_1"].values[0]
lambda_2 = params["lambda_2"].values[0]
regularization_weight = params["regularization_weight"].values[0]
entropy_weight = params["entropy_weight"].values[0]

# Model Initiation
torch.manual_seed(43)
model = Model(
    [3, len_snp, 2, 2],
    [embedding_dim, embedding_dim, embedding_dim, embedding_dim],
    attn_hidden, [mlp_hidden, mlp_hidden, mlp_hidden]
)

print("Total training data: {}".format(len(train_data)*len_snp))
total_params = sum(p.numel() for p in model.parameters())
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total Params: {}\nTotal Trainable Params: {}".format(total_params, total_trainable_params))

# Model Training
def mae_loss(prediction, target):
    return torch.sum(torch.abs(prediction - target)) / len(target)

def fit(data, n_epochs, lr, batch_size, patience_num, lambda_1, lambda_2, regularization_weight, entropy_weight, *args, **kwargs):
    patience = patience_num
    best_model, best_train_loss = None, float("inf")
    mse_per_epoch, mae_per_epoch, loss_per_epoch = [], [], []
    training_start_time = time()
    mse_loss = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
 
    model.train()
    early_stopping = 0
    for epoch in range(1, n_epochs+1): # epoch
        mse_per_batch, mae_per_batch, loss_per_batch = [], [], []
        attention_score_all_samples = []
        epoch_start_time = time()
        print("EPOCH {}".format(epoch))
        
        for _, i in enumerate(train_dataloader): # mini batch
            snp, pos, loc, var, target = i[0][0], i[0][1], i[0][2], i[0][3], i[1]
            batch_start_time = time()
            optimizer.zero_grad()
            
            # Forward Pass
            x, attn_score, y_predict = model(snp, pos, loc, var)
            attention_score_all_samples.append(attn_score)

            mse_score = mse_loss(y_predict, target)
            mse_per_batch.append(mse_score.item())
            
            penalty = RegularizationAndEntropyLoss()
#             print("MSE: ", mse_score.item())
            loss = mse_score +                 penalty(x, lambda_1, lambda_2, regularization_weight, entropy_weight, model.parameters())
#             print("Total loss: ", loss.item())
            loss_per_batch.append(loss.item())

            mae_score = mae_loss(y_predict, target)
            mae_per_batch.append(mae_score.item())
            
            print("Batch {} MSE: {}".format(_+1, mse_score.item()))
            
            # Backward Pass
            mse_score.backward()
            optimizer.step()
        
        print("MSE Epoch {}: {} ({:.3f} s/epoch)".format(epoch, mean(mse_per_batch), time()-epoch_start_time), end="\n\n")
        mse_per_epoch.append(mean(mse_per_batch))
        mae_per_epoch.append(mean(mae_per_batch))
        loss_per_epoch.append(mean(loss_per_batch))
        
        if mean(mse_per_batch) < best_train_loss:
            best_train_loss = mean(mse_per_batch)
            best_model = model
            patience = patience_num
            if epoch > 850:
                torch.save(best_model.state_dict(), "model/v3-epoch-{}-mse-{:.5f}.pt".format(epoch, best_train_loss))
        else:
            patience -= 1
        
        if patience == 0:
            print("=" * 25)
            print("BREAK POINT")
            print("The latest MSE: {}".format(best_train_loss))
            print("Early stopping at epoch {}".format(epoch))
            print("=" * 25, end="\n\n")
            early_stopping = epoch
 
    print("=" * 25)
    print("Total exc time: {:.3} s".format(time()-training_start_time))
    print("=" * 25)
        
    return attention_score_all_samples, best_model, mse_per_epoch, mae_per_epoch, loss_per_epoch, early_stopping

a, best_model, mse, mae, loss, early_stopping = fit(train_data, n_epochs=1000, lr=lr, batch_size=batch_size, lambda_1=lambda_1, lambda_2=lambda_2,
                                                    regularization_weight=regularization_weight, entropy_weight=entropy_weight, patience_num=15)

df({
    'mse':mse,
    'early_stopping':early_stopping
}).to_csv("result/training_plot/v3-with-attn1-xavier-attn2-xavier.csv")

plt.figure(figsize=(8, 5))
plt.plot(mse[1:], label="Training MSE")
plt.axvline(early_stopping, label="Early Stopping", color="r", linestyle='dashed')
plt.legend()