import torch
from collections import Counter
from ast import literal_eval as lev
from pandas import read_csv, DataFrame as df, Series
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_log_error as msle
from matplotlib import pyplot as plt
from numpy import sqrt
from ipynb.fs.full.database import load_dataset, data_processing

# Data Loading
dataset, yield_stats = load_dataset()
train_data, test_data, len_snp, max_loc, max_var = data_processing(dataset, .15)

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
#         torch.nn.init.xavier_normal_(self.attn_fc1.weight, gain=torch.nn.init.calculate_gain("relu"))
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
        fc1 = self.ReLU(self.fully_connected_1(concat.squeeze())) # convert from [dimA, dimB, 1] to [dimA, dimB]
        fc2 = self.ReLU(self.fully_connected_2(fc1))
        fc2 = fc2.unsqueeze(0)
        wide_model_1 = self.sample_location_wide_model(sample_loc)
        wide_model_2 = self.sample_variety_wide_model(sample_variety)
        wide_deep = torch.cat((wide_model_1, wide_model_2, fc2), dim=1)
        result = self.wide_deep_model(wide_deep).squeeze()
    
        return a.squeeze(), result

params = read_csv("result/v2-best_params.csv", index_col=0)
attn_hidden = int(params["attn_hidden"].values[0])
mlp_hidden = int(params["mlp_hidden"].values[0])
embedding_dim = int(params["embedding_dim"].values[0])
batch_size = int(params["batch_size"].values[0])
lr = params["lr"].values[0]

# Load Best Model
best_model = Model(
    [3, len_snp, 2, 2],
    [embedding_dim, embedding_dim, embedding_dim, embedding_dim],
    attn_hidden, [mlp_hidden, mlp_hidden, mlp_hidden]
)
best_model.load_state_dict(torch.load("model/v2-epoch-944-mse-2.29820-attn1-xavier-attn2-normal.pt"))

# Loss Functions Definition
mse_loss = torch.nn.MSELoss(reduction="mean")

def mae_loss(prediction, target):
    return torch.sum(torch.abs(prediction - target)) / len(target)

# Advanced Evaluation
def summarize(eval_model, test_data, threshold, *args, **kwargs):
    snps_name = read_csv("data/RiceToolkit/app-master/data/X.csv").columns[2:].tolist()
    snps_pos = [list(range(len(snps_name))) for i in range(len(test_data))]
    eval_model.eval() # turn on the evaluation mode
    
    counter, counter_attn = Counter(), Counter()
    for i, data in enumerate(test_data):
        test_dataloader = torch.utils.data.DataLoader(data)
        test_dataloader = list(test_dataloader)
        with torch.no_grad():
            attn, output = eval_model(test_dataloader[0][0], test_dataloader[0][1], test_dataloader[0][2], test_dataloader[0][3])
            test_mse_loss = mse_loss(output, data[1]).item()
            attn_score = attn.tolist()
            
            snp_name_dict = {}
            snp_attn_score = {}
            for j, txt in enumerate(snps_name):
                if attn_score[j] >= threshold:
                    snp_attn_score[txt] = attn_score[j]
                    snp_name_dict[txt] = 1
            counter_attn.update(snp_attn_score)
            counter.update(snp_name_dict)
    
    dataframe_1 = df(dict(counter.most_common()), index=["count"]).T # .most_common() to sort from the highest score
    dataframe_2 = df(dict(counter_attn), index=["attn_score"]).T
    dataframe = dataframe_1.merge(dataframe_2, left_index=True, right_index=True)
    dataframe.insert(0, "snp", dataframe.index)
    dataframe.set_index(Series([i for i in range(len(counter))]), inplace=True)
    dataframe = dataframe.assign(avg_attn_score=list(map(lambda x, y: x/y, dataframe["attn_score"].tolist(), dataframe["count"].tolist())))
    
    total_snps = len(dataframe.snp.tolist())
    snp_prev = ["TBGI036687_C", "TBGI050092_T", "id4009920_G", "id5014338_A", "TBGI272457_A", "id7002427_T", "id8000244_T","id10003620_T","id12006560_G"]
    intersect = list(set(dataframe.snp.tolist()) & set(snp_prev))
    snp_intersect = dataframe.set_index("snp").loc[intersect].index.to_list()
    
    return total_snps, snp_intersect, dataframe

_a, _b, snp_table = summarize(best_model, test_data, .025)
snp_table.to_csv("result/significant_snps/v2.csv")
snp_table

snp_prev = ["TBGI036687_C", "TBGI050092_T", "id4009920_G", "id5014338_A", "TBGI272457_A", "id7002427_T", "id8000244_T","id10003620_T","id12006560_G"]
intersect = list(set(snp_table.snp.tolist()) & set(snp_prev))
snp_table.set_index("snp").loc[intersect]

thresholds = [5e-3*(i+1) for i in range(1, 20)]
number_of_snps = []
for t in thresholds:
    total_snp, snp_intersect, _ = summarize(best_model, test_data, t)
    number_of_snps.append(total_snp)
    print("Threshold {} - Total SNPs: {} - Intersected SNP: {}".format(t, total_snp, snp_intersect))

# df({"threshold" : thresholds, "total_snps" : number_of_snps}).to_csv("result/significant_snps/v2-attn1-normal-attn2-xavier.csv")

def evaluate(eval_model, test_data, SAVE_PATH, start_from, *args, **kwargs):
    plot_fig = plt.figure(figsize=(15, 28))
    snps_name = read_csv("data/RiceToolkit/app-master/data/X.csv").columns[2:].tolist()
    snps_pos = [list(range(len(snps_name))) for i in range(len(test_data))]
    eval_model.eval() # turn on the evaluation mode
    
    for i, data in enumerate(test_data):
        sample_no = i + start_from
        plot_fig.add_subplot(5, 3, i+1) # position index always starts from 1, thus i+1
        test_dataloader = torch.utils.data.DataLoader(data)
        test_dataloader = list(test_dataloader)
        with torch.no_grad():
            attn, output = eval_model(test_dataloader[0][0], test_dataloader[0][1], test_dataloader[0][2], test_dataloader[0][3])
            test_mse_loss = mse_loss(output, data[1]).item()
        
            # PLOT FIGURE
            attn_score = attn.tolist()
            chr_slice = [0, 355, 440, 624, 680, 772, 928, 1012, 1058, 1097, 1128, 1181, 1232]
            color = ["dark red", "tomato", "goldenrod", "yellow", "lawn green", "lime", "dark cyan",
                     "turquoise", "blue", "dark magenta", "hot pink", "pink"]
            
            for k, v in enumerate(chr_slice[:-1]):
                plt.scatter(snps_pos[i][v:chr_slice[k+1]], attn_score[v:chr_slice[k+1]],
                    c="xkcd:{}".format(color[k]), label="Chr-{}".format(k+1))
            plt.ylim(-.1, .7)
            plt.xlabel("SNP/Chromosome")
            plt.ylabel("Attention Score")
            plt.legend(loc="upper left", fontsize="x-small")
            
            for j, txt in enumerate(snps_name):
                if attn_score[j] >= .025:
                    plt.annotate(txt, (snps_pos[i][j], attn_score[j]))

            plt.title("Sample {} (MSE Loss: {:.3f})".format(sample_no, test_mse_loss))
    plt.savefig("result/manhattan_plot/" + SAVE_PATH, bbox_inches="tight")
    plt.tight_layout() # margin adjusted

evaluate(best_model, test_data[:15], "v2-plot-(sample.no.1-15).png", 1)
evaluate(best_model, test_data[15:30], "v2-plot-(sample.no.16-30).png", 16)
evaluate(best_model, test_data[30:45], "v2-plot-(sample.no.31-45).png", 31)
evaluate(best_model, test_data[45:60], "v2-plot-(sample.no.46-60).png", 46)
evaluate(best_model, test_data[60:75], "v2-plot-(sample.no.61-75).png", 61)
evaluate(best_model, test_data[75:90], "v2-plot-(sample.no.76-90).png", 76)
evaluate(best_model, test_data[90:], "v2-plot-(sample.no.91-104).png", 91)