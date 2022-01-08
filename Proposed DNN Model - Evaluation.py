import torch
from ast import literal_eval as lev
from pandas import read_csv
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import r2_score

# Data Loading
gp_table = read_csv(r"./data/rice-gp-table.csv")
gp_table["snps"] = [lev(i) for i in gp_table["snps"]]
gp_table["snps_id"] = [lev(i) for i in gp_table["snps_id"]]

for i in range(len(gp_table.snps)):
    gp_table.snps[i].reverse()

for i in range(len(gp_table.snps_id)):
    gp_table.snps_id[i].reverse()

snp, snp_pos, loc, target = list(gp_table.snps), list(gp_table.snps_id), list(gp_table.location), list(gp_table.rice_yield)

tensor_snp = [torch.tensor(i, dtype=torch.long) for i in snp]
tensor_pos = [torch.tensor(i, dtype=torch.long) for i in snp_pos]
tensor_loc = [torch.tensor(i) for i in loc]
tensor_yield = [torch.tensor(i) for i in target]

# format: [[tensor(snp), tensor(pos), tensor(loc)], tensor(yield)]
dataset = [data for data in zip(zip(tensor_snp, tensor_pos, tensor_loc), tensor_yield)]

# Data split scheme (train : val) : test = (70 : 15) : 15
train_data, test_data = tts(dataset, test_size=.15, shuffle=True, random_state=43)

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
        torch.nn.init.uniform_(self.attn_fc1.weight)
        torch.nn.init.uniform_(self.attn_fc2.weight)
        torch.nn.init.uniform_(self.attn_fc1.bias)
        torch.nn.init.uniform_(self.attn_fc2.bias)
    
    def forward(self, x, *args, **kwargs):
        x = self.ReLU(self.attn_fc1(x))
        x = self.softmax(self.attn_fc2(x))
        return x

class Model(torch.nn.Module):
    def __init__(self, num_embedding, embedding_dim, attn_hidden, mlp_hidden, *args, **kwargs):
        super(Model, self).__init__()
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.attn_hidden = attn_hidden
        self.mlp_hidden = mlp_hidden
        
        # Embedding for SNP
        self.input_embedding = torch.nn.Embedding(self.num_embedding[0], self.embedding_dim[0])
        
        # Embedding for SNP position
        self.positional_embedding = torch.nn.Embedding(self.num_embedding[1], self.embedding_dim[1])
        
        # Embedding for sample location
        self.loc_embedding = torch.nn.Embedding(self.num_embedding[2], self.embedding_dim[2])
        
        self.one_hot = torch.nn.functional.one_hot
        self.flatten = torch.flatten
        self.deep = self.mlp_hidden[1]
        self.wide = self.mlp_hidden[2]
        self.attn = AttentionMechanism(self.embedding_dim[0], self.attn_hidden)
        self.concat = torch.nn.AvgPool2d((1, self.embedding_dim[0]), stride=1) # kernel_size == embedding_dim; padding==0==same
        self.fully_connected_1 = torch.nn.Linear(self.num_embedding[1], self.mlp_hidden[0])
        self.fully_connected_2 = torch.nn.Linear(self.mlp_hidden[0], self.mlp_hidden[1])
        self.fully_connected_3 = torch.nn.Linear(3 * self.embedding_dim[2], self.mlp_hidden[2])
        self.wide_deep_model = torch.nn.Linear(self.wide+self.deep, 1)
        self.ReLU = torch.nn.ReLU()
        self.init_weights()

    def init_weights(self, *args, **kwargs):
        torch.nn.init.normal_(self.input_embedding.weight)
        torch.nn.init.normal_(self.positional_embedding.weight)
        torch.nn.init.normal_(self.loc_embedding.weight)
        torch.nn.init.normal_(self.fully_connected_1.weight)
        torch.nn.init.normal_(self.fully_connected_2.weight)
        torch.nn.init.normal_(self.fully_connected_1.bias)
        torch.nn.init.normal_(self.fully_connected_2.bias)

    def forward(self, snp, snp_pos, snp_loc, *args, **kwargs):
#         dict_ = {}
        
        inp_1 = self.input_embedding(snp) # * sqrt(self.embedding_dim[0])
#         dict_["snp_embedding"] = str(list(inp_1.size()))
#         print("input: ", inp_1.size())
        
        inp_2 = self.positional_embedding(snp_pos) # * sqrt(self.embedding_dim[1])
#         dict_["positional_embedding"] = str(list(inp_2.size()))
#         print("pos: ", inp_2)
#         print("pos: ", inp_2.size())
        
        x = inp_1 + inp_2
#         dict_["snp_and_position"] = str(list(x.size()))
#         print("input+pos: ", x.size())
        
        a = self.attn(x)
#         dict_["attention"] = str(list(a.size()))
#         print("a: ", a.squeeze())
        print("attn (softmax): ", a.size())
        
        context_vect = x*a
#         dict_["context_vect"] = str(list(context_vect.size()))
#         print("context vector: ", context_vect.size())
        
        concat = self.concat(context_vect)
#         dict_["concat_gap"] = str(list(concat.size()))
#         print("concat: ", concat.size())
        
        fc1 = self.fully_connected_1(concat.squeeze()) # convert from [dimA, dimB, 1] to [dimA, dimB]
        fc1 = self.ReLU(fc1)
#         dict_["deep_model_fc1"] = str(list(fc1.size()))
#         print("fc1: ", fc1.size())
        
        fc2 = self.fully_connected_2(fc1)
        deep_model = self.ReLU(fc2)
#         dict_["deep_model"] = str(list(deep_model.size()))
#         print("deep_model: ", deep_model.size())
        
        snp_loc = self.one_hot(snp_loc)
#         dict_["snp_loc_one_hot"] = str(list(snp_loc.size()))
#         print("onehot: ", snp_loc.size())

        snp_loc = self.loc_embedding(snp_loc)
#         dict_["snp_loc_embedding"] = str(list(snp_loc.size()))
#         print("embed: ", snp_loc.size())

        snp_loc_flatten = self.flatten(snp_loc, start_dim=1)
#         dict_["snp_loc_flatten"] = str(list(snp_loc_flatten.size()))
        
        wide_model = self.fully_connected_3(snp_loc_flatten)
#         dict_["wide_model"] = str(list(wide_model.size()))
#         print("wide_model: ", wide_model.size())
        
        wide_deep = torch.cat((wide_model, deep_model), dim=1)
#         dict_["wide_deep_concat"] = str(list(wide_deep.size()))
#         print("wide_deep_concat: ", wide_deep.size())
        
        result = self.wide_deep_model(wide_deep)
#         dict_["wide_deep_model"] = str(list(result.size()))
        
        result = result.squeeze()
#         dict_["final_output"] = str(list(result.size()))
#         print("result: ", result.size())
        
#         layer_size = df(dict_, index=["layer_size"]).T
#         layer_size.to_csv("layer_size.csv", index=False)
    
        return a.squeeze(), result


# Load Best Model
best_model = Model([3, 142, 3], [64, 64, 64], 32, [64, 32, 32])
best_model.load_state_dict(torch.load("model/best-model-params-epoch-292-mse-2.64629.pt"))

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
            print(attn[0])
            test_mse_loss = mse_loss(output, data[1]).item()
            test_mae_loss = mae_loss(output, data[1]).item()
            test_r2_loss = r2_score(data[1].tolist(), output.tolist()).item()
            test_adj_r2_loss = 1-(1-test_r2_loss)*(n-1)/(n-p-1)

            print("MSE Loss: {:.3f}".format(test_mse_loss))
            print("MAE Loss: {:.3f}".format(test_mae_loss))
            print("R2: {:.3f}".format(test_r2_loss))
            print("Adj. R2: {:.3f}".format(test_adj_r2_loss))

evaluate(best_model, test_dataloader)