import os
import numpy as np
import subprocess
import torch
import torch.utils.data as Data
from torch import nn
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from CNNmodel import CAPCNN
import argparse
import pickle as pkl



"""
===============================================================
                        Input Params
===============================================================
"""
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--gpus', type=int, default = 0)
parser.add_argument('--n', type=int, default=218)
parser.add_argument('--kmers', type=str, default='3,7,11,15')
parser.add_argument('--t', type=float, default=0.6)
parser.add_argument('--embed', type=str, default="Embed.pkl")
parser.add_argument('--classifier', type=str, default="Params.pkl")
parser.add_argument('--rejection', type=str, default="N")
parser.add_argument('--outpath', type=str, default="result")
args = parser.parse_args()

kmers = args.kmers
kmers = kmers.split(',')
kmers = list(map(int, kmers))

if not os.path.exists(f"{args.outpath}/CNN_Classifier/"):
    _ = subprocess.check_call(f"cp -r CNN_Classifier/ {args.outpath}/CNN_Classifier/", shell=True)
try:
    os.chdir(f"{args.outpath}/CNN_Classifier/")
except:
    print("no CNN classifier is avaliable")
    exit(1)

contig_in = "../input/"
contig_out = "validation/"
Knowledge_graph = "Cyber_data/"

if not os.path.exists(Knowledge_graph):
    _ = os.makedirs(Knowledge_graph)
else:
    print("folder {0} exist... cleaning dictionary".format(Knowledge_graph))
    if os.listdir(Knowledge_graph):
        try:
            _ = subprocess.check_call("rm -rf {0}".format(Knowledge_graph), shell=True)
            _ = os.makedirs(Knowledge_graph)
            print("Dictionary cleaned")
        except:
            print("Cannot clean your folder... permission denied")
            exit(1)

#try:
#    _ = subprocess.check_call("mkdir {0}".format(Knowledge_graph), shell=True)
#except:
#    try:
#        _ = subprocess.check_call("rm {0}*".format(Knowledge_graph), shell=True)
#    except:
#        print("Cannot clean your folder... permission denied")

_ = os.system("bash clean_all_script.sh")
file_list = sorted(os.listdir(contig_in))
seq = []
old_file_id = 0
contig_id = 0
with open("name_list.csv",'w') as list_out:
    list_out.write("contig_name,idx\n")
    for file_n in file_list:
        for record in SeqIO.parse(contig_in+file_n, "fasta"):
            name = str(old_file_id) + "_" + str(contig_id)
            contig_id += 1
            list_out.write(record.id + "," + name)
            _ = SeqIO.write(record, contig_out+name+".fasta", "fasta")
        old_file_id += 1

try:
    print("Capturing compressed features")
    _ = subprocess.check_call("bash code/compress_script.sh", shell=True)
except:
    print("Script error")
    exit(1)


"""
===========================================================
                Load Trained Model
===========================================================
"""

if torch.cuda.is_available():
    torch.cuda.set_device(args.gpus)
else:
    print("Running with cpu")

cnn = CAPCNN.WCNN(num_token=100,num_class=args.n,kernel_sizes=kmers, kernel_nums=[256, 256, 256, 256])
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
compress_feature = []
compress_label = []

        
file_list = os.listdir("dataset")
file_list = sorted(file_list)

for name in file_list:
    val = np.genfromtxt('dataset/'+name, delimiter=',')
    val_label = val[:, -1]
    val_feature = val[:, :-1]
    label = int(name.split(".")[0])
    # comvert format
    val_feature = torch.from_numpy(val_feature).long()
    val_label = torch.from_numpy(val_label).float()
    val_feature = torch_embeds(val_feature)
    val_feature = val_feature.reshape(len(val_feature), 1, 1698, 100)
    # prediction
    if torch.cuda.is_available():
        with torch.no_grad():
            out = cnn(val_feature.cuda())
        out = out.cpu().detach().numpy()  
    else:
        with torch.no_grad():
            out = cnn(val_feature)
        out = out.detach().numpy()

    out = np.sum(out, axis=0)
    compress_feature.append(out)
    compress_label.append(label)



compress_feature = np.array(compress_feature)
pkl.dump(compress_feature, open("../Cyber_data/contig.F", 'wb'))
_ = os.system("bash clean_all_script.sh")
