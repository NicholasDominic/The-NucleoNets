import torch
from ast import literal_eval as lev
from pandas import read_csv
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import r2_score
from ipynb.fs.full.neural_net import load_dataset, data_processing

# Data Loading
dataset, yield_stats = load_dataset()
train_data, test_data, len_snp, loc = data_processing(dataset, .15)

# Model Class
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
            data = self.one_hot(data, num_classes=max(variety)+1)
        else:
            data = self.one_hot(data)
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
#         print("input: ", snp.size())
        
        snp_pos = self.positional_embedding(snp_pos) # * sqrt(self.embedding_dim[1])
#         print("pos: ", snp_pos.size())
        
        x = snp + snp_pos
#         print("input+pos: ", x.size())
        
        a = self.attn(x)
#         print("attn (softmax): ", a.size())
        
        context_vect = x*a
#         print("context vector: ", context_vect.size())
        
        concat = self.concat(context_vect)
#         print("concat: ", concat.size())
        
        fc1 = self.ReLU(self.fully_connected_1(concat.squeeze())) # convert from [dimA, dimB, 1] to [dimA, dimB]
#         print("fc1: ", fc1.size())
        
        fc2 = self.ReLU(self.fully_connected_2(fc1))
#         print("fc2: ", fc2.size())
    
        wide_model_1 = self.sample_location_wide_model(sample_loc)
#         print("wide_model_1: ", wide_model_1.size())
        
#         wide_model_2 = self.sample_variety_wide_model(sample_variety)
        wide_deep = torch.cat((wide_model_1, fc2), dim=1)
#         print("wide_deep_concat: ", wide_deep.size())
        
        result = self.wide_deep_model(wide_deep).squeeze()
#         print("result: ", result.size())
    
        return a.squeeze(), result

params = read_csv("result/v6-best_params.csv", index_col=0)
attn_hidden = int(params["attn_hidden"].values[0])
mlp_hidden = int(params["mlp_hidden"].values[0])
embedding_dim = int(params["embedding_dim"].values[0])
batch_size = int(params["batch_size"].values[0])
lr = params["lr"].values[0]

# Load Best Model
best_model = Model(
    [3, len_snp, 2, 2],
    [embedding_dim, embedding_dim, embedding_dim, 0],
    attn_hidden, [mlp_hidden, mlp_hidden, mlp_hidden]
)
best_model.load_state_dict(torch.load("model/v6-best-model-params-regularization-entropy-epoch-41-mse-2.55585.pt"))

# Loss Functions Definition
mse_loss = torch.nn.MSELoss(reduction="mean")
def mae_loss(prediction, target):
    return torch.sum(torch.abs(prediction - target)) / len(target)

# Evaluation
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))

def evaluate(eval_model, data, *args, **kwargs):
    eval_model.eval() # turn on the evaluation mode
    n, p = len(test_data), 2 # for Adjusted R2 score, n = number of samples, p = number of independent variable
    
    for data in test_dataloader:
        with torch.no_grad():
            attn, output = eval_model(data[0][0], data[0][1], data[0][2])
            test_mse_loss = mse_loss(output, data[1]).item()
            test_mae_loss = mae_loss(output, data[1]).item()
            test_r2_loss = r2_score(data[1].tolist(), output.tolist()).item()
            test_adj_r2_loss = 1-(1-test_r2_loss)*(n-1)/(n-p-1)

            print("MSE Loss: {:.3f}".format(test_mse_loss))
            print("MAE Loss: {:.3f}".format(test_mae_loss))
            print("R2: {:.3f}".format(test_r2_loss))
            print("Adj. R2: {:.3f}".format(test_adj_r2_loss))

evaluate(best_model, test_dataloader)