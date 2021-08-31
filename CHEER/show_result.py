import numpy as np
import torch
import torch.utils.data as Data
from torch import nn
from model import Wcnn
import argparse



"""
===============================================================
                        Input Params
===============================================================
"""

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--gpus', type=int, default = 1)
parser.add_argument('--n', type=int, default=2)
parser.add_argument('--kmers', type=str, default='3,7,11,15')
parser.add_argument('--t', type=float, default=0.6)
parser.add_argument('--embed', type=str, default="embed.pkl")
parser.add_argument('--classifier', type=str, default="Reject_params.pkl")
parser.add_argument('--rejection', type=str, default="N")
args = parser.parse_args()

kmers = args.kmers
kmers = kmers.split(',')
kmers = list(map(int, kmers))


"""
===========================================================
                Load Trained Model
===========================================================
"""
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpus)
else:
    print("Running with cpu")

cnn = Wcnn.WCNN(num_token=100,num_class=args.n,kernel_sizes=kmers, kernel_nums=[256, 256, 256, 256])
#cnn = Wcnn.WCNN(num_token=100,num_class=20,kernel_sizes=[3, 7, 11, 15], kernel_nums=[256, 256, 256, 256], seq_len=244)
pretrained_dict=torch.load(args.classifier, map_location='cpu')
cnn.load_state_dict(pretrained_dict)

# Evaluation mode
cnn = cnn.eval()
if torch.cuda.is_available():
    cnn = cnn.cuda()

# Load embedding
torch_embeds = nn.Embedding(65, 100)
tmp = torch.load(args.embed, map_location='cpu')
old_weight = tmp['weight']
padding = torch.zeros((1, 100))
new_weight = torch.cat([tmp['weight'], padding])
torch_embeds.weight = torch.nn.Parameter(new_weight)
torch_embeds.weight.requires_grad=False


"""
===========================================================
                Load Validation Dataset
===========================================================
"""

val = np.genfromtxt('dataset/val.csv', delimiter=',')
#val = np.genfromtxt('dataset/family_validation.csv', delimiter=',')
val_label = val[:, -1]
val_feature = val[:, :-1]

val_feature = torch.from_numpy(val_feature).long()
val_label = torch.from_numpy(val_label).float()
val_feature = torch_embeds(val_feature)
if val_feature.reshape(-1).shape[0] == 24800:
    val_feature = val_feature.reshape(1, 1, 248, 100)
else:
    val_feature = val_feature.reshape(len(val_feature), 1, 248, 100)


"""
===========================================================
                Record Confusion Matrix
===========================================================
"""
def softmax(x):
    return np.exp(x)/sum(np.exp(x))

with open("prediction/result.txt", 'w') as file:
    idx = 1
    prediction = []
    with torch.no_grad():
        for (feature, label) in zip(val_feature, val_label):
            if torch.cuda.is_available():
                pred = cnn(torch.unsqueeze(feature.cuda(), 0))
                pred = pred.cpu().detach().numpy()[0]
            else:
                pred = cnn(torch.unsqueeze(feature, 0))
                pred = pred.detach().numpy()[0]
            pred = softmax(pred)
            if max(pred) > args.t:
                y = int(np.argmax(pred))
                file.write(str(idx) + "->" + str(y+1) +"\n")
            else:
                if args.rejection == "Y":
                    file.write(str(idx) + "->" + str(2) +"\n")
                else:
                    file.write(str(idx) + "->" + str(0) +"\n")
            idx+=1

