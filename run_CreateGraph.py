import os
import sys
import Bio
import logging
import argparse
import subprocess
import scipy as sp
import numpy as np
import pandas as pd
import pickle as pkl
import networkx as nx
import scipy.stats as stats
import scipy.sparse as sparse
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


# Defined folder
out_f = "out/"
contig_in = "input/"
contig_out = "single_contig/"
file_in_fn = "single_contig/"
file_out_fn = "all_proteins/"
Knowledge_graph = "Cyber_data/"
all_protein_f = out_f+"all_translate_proteins.fa"
proteins_aa_fp = all_protein_f
db_fp = "database/database.dmnd"
diamond_out_fn = '{}.diamond.tab'.format(os.path.basename(proteins_aa_fp).rsplit('.', 1)[0])
diamond_out_fp = os.path.join(out_f, diamond_out_fn)
contig_abc_fp = out_f + diamond_out_fn + ".abc"
abc_fp = out_f+"merged.abc"


# Generating E-edges
print("\n\n" + "{:-^80}".format("Calculating E-edges"))

# loading database
database = "database/"
gene2genome = pd.read_csv(database+"ALL_gene_to_genomes.csv")
contig_id = gene2genome["contig_id"].values
contig_id = [item.replace(" ", "~") for item in contig_id]
gene2genome["contig_id"] = contig_id
#taxnomic_df = pd.read_csv(database+"merged_df.csv", )
#taxnomic_df = taxnomic_df.drop(["Unnamed: 0", 'pos'], axis=1)

protein_to_ref = {protein:ref for protein, ref in zip(gene2genome["protein_id"].values, gene2genome["contig_id"].values)}
#ref_to_num = {ref:num for ref, num in zip(taxnomic_df["contig_id"].values, taxnomic_df["proteins"].values)}

contig_set = list(set(gene2genome["contig_id"].values))
ID_to_ref = {i:ref for i, ref in enumerate(contig_set)}
ref_to_ID = {ref:i for i, ref in enumerate(contig_set)}

fn = "single_contig/"
contig_to_id = {}
file_list = os.listdir(fn)
file_list = sorted(file_list)
for file_n in file_list:
    name = file_n.split(".")[0]
    contig_to_id[name] = file_list.index(file_n)


# record the row id for each contigs
id_to_contig = {value: key for key, value in contig_to_id.items()}

fn = "out/"
blastp = pd.read_csv(contig_abc_fp, sep=" ", names = ["contigs", "ref", "e-value"])
gene_to_genome = pd.read_csv(fn+"contig_gene_to_genome.csv", sep=",")

e_matrix = np.ones((len(contig_to_id), len(ref_to_ID.keys())))
blast_contigs = blastp["contigs"].values
blast_ref = blastp["ref"].values
blast_value = blastp["e-value"].values
for i in range(len(blast_contigs)):
    contig_name = gene_to_genome[gene_to_genome["protein_id"] == blast_contigs[i]]["contig_id"].values
    contig_name = contig_name[0]
    row_id = contig_to_id[contig_name]
    reference = protein_to_ref[blast_ref[i]]
    col_id = ref_to_ID[reference]
    e_value = float(blast_value[i])
    if e_value == 0:
        e_value = 1e-250
    if e_matrix[row_id][col_id] == 1:
        e_matrix[row_id][col_id] = e_value
    else:
        e_matrix[row_id][col_id] += e_value

e_weight = -np.log10(e_matrix)-50
e_weight[e_weight < 1] = 0


# Generating P-edges
print("\n\n" + "{:-^80}".format("Calculating P-edges"))
database = "database/"

name_to_id = {}
reference_df = pd.read_csv("database/reference_name_id.csv")
tmp_ref = reference_df["name"].values
tmp_id  = reference_df["idx"].values
for ref, idx in zip(tmp_ref,tmp_id):
    name_to_id[ref.replace(" ", "~")] = idx

#file_list = os.listdir(database+"species/")
#file_list = sorted(file_list)
#for file_n in file_list:
#    idx = file_n.split(".")[0]
#    for record in SeqIO.parse(database+"species/"+file_n, "fasta"):
#        name = record.description
#        name = name.split("|")[1]
#        name = name.split(",")[0]
#        name = name.replace(" ", "~")
#    name_to_id[name] = idx





edges = pd.read_csv(out_f+"network.ntw", sep=' ', names=["node1", "node2", "weight"])
merged_df = pd.read_csv(database+"ALL_genome_profile.csv", header=0, index_col=0)
Taxonomic_df = pd.read_csv(database+"taxonomic_label.csv")
merged_df = pd.merge(merged_df, Taxonomic_df, left_on="contig_id", right_on="contig_id", how="inner")
contig_id = merged_df["contig_id"].values
family = merged_df["class"].values
contig_to_family = {name: family for name, family in zip(contig_id, family) if type(family) != type(np.nan) }

G = nx.Graph()
# Add p-edges to the graph
with open(out_f+"/network.ntw") as file_in:
    for line in file_in.readlines():
        tmp = line[:-1].split(" ")
        node1 = tmp[0]
        node2 = tmp[1]
        weight = float(tmp[2])
        
        #if node1 == "Gordonia~phage~GMA6" or node2 == "Gordonia~phage~GMA6":
        #    continue
        #if node1 == "Acinetobacter~phage~vB_AbaM_ME3" or node2 == "Acinetobacter~phage~vB_AbaM_ME3":
        #    continue
        
        if "~" in node1 and node1 not in name_to_id.keys():
            print(node1)
            print("ERROR")
            exit(1)
        if "~" in node2 and node2 not in name_to_id.keys():
            print(node2)
            print("ERROR")
            exit(1)

        G.add_edge(node1, node2, weight = 1)
        #if node1 in name_to_id.keys() and node2 in name_to_id.keys():
        #    G.add_edge(node1, node2, weight = 1)
        #elif "_" in node1 and node2 in name_to_id.keys():
        #    G.add_edge(node1, node2, weight = 1)
        #elif node1 in name_to_id.keys() and "_" in node2 :
        #    G.add_edge(node1, node2, weight = 1)
        #elif "_" in node1 and "_" in node2 :
        #    G.add_edge(node1, node2, weight = 1)    
        #else:
        #    continue

# Add e-edges to the graph
cnt = 0
for i in range(e_weight.shape[0]):
    contig_name = id_to_contig[i]
    if contig_name not in G.nodes():
        sorted_idx = np.argsort(e_weight[i])
        for j in range(5):
            idx = sorted_idx[-j]
            if e_weight[i][idx] != 0:
                ref_name = ID_to_ref[idx]
                if ref_name in G.nodes():
                    G.add_edge(contig_name, ref_name, weight = 1)
                    cnt += 1

# remove the uncompressed nodes
node_list = list(G.nodes())
for node in node_list:
    if "~" in node and node not in contig_to_family.keys():
        G.remove_node(node)

test_to_id = {}
class_to_label = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 5, 9: 5, 10: 5, 11: 5, 12: 5, 13: 5, 14: 6, 15: 6, 16: 6, 17: 7, 18: 7, 19: 7, 20: 7, 21: 7, 22: 7, 23: 7, 24: 7, 25: 7, 26: 7}

# Generating the Knowledge Graph
print("\n\n" + "{:-^80}".format("Generating Knowledge graph"))
mode = "testing"
if mode == "validation":
    test_mask = []
    label = []
    cnt = 0
    for node in G.nodes():
        try:
            label.append(class_to_label[contig_to_family[node]])
            cnt+=1
        except:
            if "_" in node:
                try:
                    class_ = int(node.split("_")[0])
                    label.append(class_)
                    test_mask.append(cnt)
                    test_to_id[node] = cnt
                    cnt+=1
                except:
                    print(node)
            else:
                print(node)
    pkl.dump(test_mask, open("Cyber_data/contig.mask", "wb" ) )
    pkl.dump(label, open("Cyber_data/contig.label", "wb" ) )
    adj = nx.adjacency_matrix(G)
    pkl.dump(adj, open("Cyber_data/contig.graph", "wb" ) )
    pkl.dump(test_to_id, open("Cyber_data/contig.dict", "wb" ) )



if mode == "testing":
    test_mask = []
    label = []
    cnt = 0
    for node in G.nodes():
        try:
            label.append(class_to_label[contig_to_family[node]])
            cnt+=1
        except:
            if "_" in node:
                try:
                    label.append(-1)
                    test_mask.append(cnt)
                    test_to_id[node] = cnt
                    cnt+=1
                except:
                    print(node)
            else:
                print(node)
    pkl.dump(test_mask, open("Cyber_data/contig.mask", "wb" ) )
    adj = nx.adjacency_matrix(G)
    pkl.dump(adj, open("Cyber_data/contig.graph", "wb" ) )
    pkl.dump(test_to_id, open("Cyber_data/contig.dict", "wb" ) )


# contructing feature map
fn = "database"
contig_feature = pkl.load(open("Cyber_data/contig.F",'rb'))
database_feature = pkl.load(open(fn+"/dataset_compressF",'rb'))

feature = []
for node in G.nodes():
    if "~" not in node:
        idx = contig_to_id[node]
        feature.append(contig_feature[idx])
    else:
        try:
            idx = int(name_to_id[node])
            feature.append(database_feature[idx])
        except:
            print(node)

feature = np.array(feature)
if mode == "testing":
    pkl.dump(feature, open("Cyber_data/contig.feature", "wb" ) )
else:
    pkl.dump(feature, open("Cyber_data/contig.feature", "wb" ) )


# Graph check for each testing samples
cnt = 0
for node in G.nodes:
    if "~" not in node:
        neighbor_label = []
        for edge in G.edges(node):
            neighbor = edge[1]
            if "~" in neighbor:
                neighbor_label.append(class_to_label[contig_to_family[neighbor]])
            else:
                continue
        if len(set(neighbor_label)) == 1:
            label[test_to_id[node]] = neighbor_label[0]
            cnt += 1

pkl.dump(label, open("Cyber_data/contig.label", "wb" ) )