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

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--n', type=int, default = 0)
args = parser.parse_args()
################################################################################
############################  Check the folder #################################
################################################################################
if not os.path.exists(out_f):
    _ = os.makedirs(out_f)
else:
    print("folder {0} exist... cleaning dictionary".format(out_f))
    if os.listdir(out_f):
        try:
            _ = subprocess.check_call("rm -rf {0}".format(out_f), shell=True)
            _ = os.makedirs(out_f)
            print("Dictionary cleaned")
        except:
            print("Cannot clean your folder... permission denied")
            exit(1)

if not os.path.exists(file_in_fn):
    _ = os.makedirs(file_in_fn)
else:
    print("folder {0} exist... cleaning dictionary".format(file_in_fn))
    if os.listdir(file_in_fn):
        try:
            _ = subprocess.check_call("rm -rf {0}".format(file_in_fn), shell=True)
            _ = os.makedirs(file_in_fn)
            print("Dictionary cleaned")
        except:
            print("Cannot clean your folder... permission denied")
            exit(1)

if not os.path.exists(file_out_fn):
    _ = os.makedirs(file_out_fn)
else:
    print("folder {0} exist... cleaning dictionary".format(file_out_fn))
    if os.listdir(file_out_fn):
        try:
            _ = subprocess.check_call("rm -rf {0}".format(file_out_fn), shell=True)
            _ = os.makedirs(file_out_fn)
            print("Dictionary cleaned")
        except:
            print("Cannot clean your folder... permission denied")
            exit(1)



################################################################################
############################  Rename the files  ################################
################################################################################

# Split contigs files into single contig per file.
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
            list_out.write(record.id + "," + name + "\n")
            _ = SeqIO.write(record, contig_out+name+".fasta", "fasta")
        old_file_id += 1


################################################################################
###################### Translate contigs into 6 ORFs ###########################
################################################################################

def return_protien(contig):
    frame = Bio.Seq.translate(contig)
    protein_list = frame.split("*")
    sorted_list = sorted(protein_list, key=len, reverse=True)
    sorted_list = [item for item in sorted_list if len(item) > 10 ]
    return sorted_list


file_list = os.listdir(file_in_fn)
for file in file_list:
    old_file_name = file.rsplit(".", 1)[0]
    contig_id = int(old_file_name.split("_")[-1])
    label_id = int(old_file_name.split("_")[0])
    for record in SeqIO.parse(file_in_fn+file, "fasta"):
        protein_record = []
        contig = str(record.seq)
        # conver into protein
        frame1 = return_protien(contig)
        frame2 = return_protien(contig[1:])
        frame3 = return_protien(contig[2:])
        rev_contig = Bio.Seq.reverse_complement(contig)
        frame4 = return_protien(rev_contig)
        frame5 = return_protien(rev_contig[1:])
        frame6 = return_protien(rev_contig[2:])
        proteins = np.concatenate([frame1, frame2, frame3, frame4, frame5, frame6])
        for i in range(len(proteins)):
            rec = SeqRecord(Seq(proteins[i]), id=str(label_id)+ "_" + str(contig_id) + "_" + str(i), description="")
            protein_record.append(rec)
        _ = SeqIO.write(protein_record, file_out_fn+file, "fasta")

all_protein_f = out_f+"all_translate_proteins.fa"
_ = subprocess.check_call("cat {0} > {1}".format(file_out_fn+"*", all_protein_f), shell=True)


################################################################################
############################## Run diamond BLASTp  #############################
################################################################################

print("\n\n" + "{:-^80}".format("Diamond BLASTp"))
print("Creating Diamond database and running Diamond...")

def make_diamond_db(cpu: int):
    diamond_db_bp = "database/database.dmnd"
    aa_fp = "database/ALL_protein.fasta"

    make_diamond_cmd = ['diamond', 'makedb', '--threads', str(cpu), '--in', aa_fp, '-d', diamond_db_bp]
    print("Creating Diamond database...")
    res = subprocess.run(make_diamond_cmd, check=True, stdout=subprocess.PIPE)
    if res.returncode != 0:
        print('Error creating Diamond database')
        exit(1)
    diamond_db_fp = diamond_db_bp + '.dmnd'
    return diamond_db_fp


def run_diamond(aa_fp, db_fp, cpu: int, diamond_out_fn):
    # More sensitive as an option?
    diamond_cmd = ['diamond', 'blastp', '--threads', str(cpu), '--sensitive', '-d', db_fp, '-q', aa_fp,
                   '-o', diamond_out_fn]
    print("Running Diamond...")
    res = subprocess.run(diamond_cmd, check=True, stdout=subprocess.PIPE)
    if res.returncode != 0:
        print('Error running Diamond')
        exit(1)
    return diamond_out_fn

proteins_aa_fp = all_protein_f
db_fp = "database/database.dmnd"
diamond_out_fn = '{}.diamond.tab'.format(os.path.basename(proteins_aa_fp).rsplit('.', 1)[0])
diamond_out_fp = os.path.join(out_f, diamond_out_fn)
# Create database
_ = make_diamond_db(64)
# Run BLASTP
similarity_fp = run_diamond(proteins_aa_fp, db_fp, 64, diamond_out_fp)

# capture the query, referencde, e-value from diamond output
contig_abc_fp = out_f + diamond_out_fn + ".abc"
abc_fp = out_f+"merged.abc"
_ = subprocess.check_call("awk '$1!=$2 {{print $1,$2,$11}}' {0} > {1}".format(diamond_out_fp, contig_abc_fp), shell=True)
_ = subprocess.check_call("cat database/database.self-diamond.tab.abc {0} > {1}".format(contig_abc_fp, abc_fp), shell=True)

# Generating gene-to-genome.csv: protein_id, contig_id, keywords
blastp = pd.read_csv(contig_abc_fp, sep=' ', names=["contig", "ref", "e-value"])
protein_id = sorted(list(set(blastp["contig"].values)))
contig_id = [item.rsplit("_", 1)[0] for item in protein_id]
description = ["hypothetical protein" for item in protein_id]
gene2genome = pd.DataFrame({"protein_id": protein_id, "contig_id": contig_id ,"keywords": description})
gene2genome.to_csv(out_f+"contig_gene_to_genome.csv", index=None)

# Counting for the mapped proteins
mapped_record = []
for record in SeqIO.parse(all_protein_f, "fasta"):
    if record.id in protein_id:
        mapped_record.append(record)
        
# Save the mapped proteins
mapped_fp = out_f+"mapped_protein.fa"
with open(mapped_fp, 'w') as file_out:
    _ = SeqIO.write(mapped_record, file_out, "fasta")


# Combining the gene-to-genomes files
_ = subprocess.check_call("cat database/ALL_gene_to_genomes.csv {0}contig_gene_to_genome.csv > {1}gene_to_genome.csv".format(out_f, out_f), shell=True)



# Run MCL
print("\n\n" + "{:-^80}".format("Protein clustering"))
print("Loading proteins...")
gene2genome_fp = out_f+"gene_to_genome.csv"
gene2genome_df = pd.read_csv(gene2genome_fp, sep=',', header=0)
def make_protein_clusters_mcl(abc_fp, out_p, inflation=2):
    """
    Args: 
        blast_fp (str): Path to blast results file
        inflation (float): MCL inflation value
        out_p (str): Output directory path
    Returns:
        str: fp for MCL clustering file
    """
    #logger.debug("Generating abc file...")
    #blast_fn = os.path.basename(blast_fp)
    #abc_fn = '{}.abc'.format(blast_fn)
    #abc_fp = os.path.join(out_p, abc_fn)
    #subprocess.check_call("awk '$1!=$2 {{print $1,$2,$11}}' {0} > {1}".format(blast_fp, abc_fp), shell=True)
    print("Running MCL...")
    abc_fn = "merged"
    mci_fn = '{}.mci'.format(abc_fn)
    mci_fp = os.path.join(out_p, mci_fn)
    mcxload_fn = '{}_mcxload.tab'.format(abc_fn)
    mcxload_fp = os.path.join(out_p, mcxload_fn)
    subprocess.check_call("mcxload -abc {0} --stream-mirror --stream-neg-log10 -stream-tf 'ceil(200)' -o {1}"
                          " -write-tab {2}".format(abc_fp, mci_fp, mcxload_fp), shell=True)
    mcl_clstr_fn = "{0}_mcl{1}.clusters".format(abc_fn, int(inflation*10))
    mcl_clstr_fp = os.path.join(out_p, mcl_clstr_fn)
    subprocess.check_call("mcl {0} -I {1} -use-tab {2} -o {3}".format(
        mci_fp, inflation, mcxload_fp, mcl_clstr_fp), shell=True)
    return mcl_clstr_fp

pc_overlap, pc_penalty, pc_haircut, pc_inflation = 0.8, 2.0, 0.1, 2.0
pcs_fp = make_protein_clusters_mcl(abc_fp, out_f, pc_inflation)



def load_mcl_clusters(fi):
    """
    Load given clusters file
    
    Args:
        fi (str): path to clusters file
        proteins_df (dataframe): A dataframe giving the protein and its contig.
    Returns: 
        tuple: dataframe proteins and dataframe clusters
    """
    # Read MCL
    with open(fi) as f:
        c = [line.rstrip("\n").split("\t") for line in f]
    c = [x for x in c if len(c) > 1]
    nb_clusters = len(c)
    formatter = "PC_{{:>0{}}}".format(int(round(np.log10(nb_clusters))+1))
    name = [formatter.format(str(i)) for i in range(nb_clusters)]
    size = [len(i) for i in c]
    clusters_df = pd.DataFrame({"size": size, "pc_id": name}).set_index("pc_id")
    return clusters_df, name, c


def build_clusters(fp, gene2genome):
    """
        Build clusters given clusters file

        Args:
            fp (str): filepath of clusters file
            gene2genome (dataframe): A dataframe giving the protein and its genome.
            mode (str): clustering method
        Returns:
            tuple: dataframe of proteins, clusters, profiles and contigs
        """
    # Read MCL
    clusters_df, name, c = load_mcl_clusters(fp)
    print("Using MCL to generate PCs.")
    # Assign each prot to its cluster
    gene2genome.set_index("protein_id", inplace=True)  # id, contig, keywords, cluster
    for prots, clust in zip(c, name):
        try:
            gene2genome.loc[prots, "cluster"] = clust
        except KeyError:
            prots_in = [p for p in prots if p in gene2genome.index]
            not_in = frozenset(prots) - frozenset(prots_in)
            print("{} protein(s) without contig: {}".format(len(not_in), not_in))
            gene2genome.loc[prots_in, "cluster"] = clust
    # Keys
    for clust, prots in gene2genome.groupby("cluster"):
        clusters_df.loc[clust, "annotated"] = prots.keywords.count()
        if prots.keywords.count():
            keys = ";".join(prots.keywords.dropna().values).split(";")
            key_count = {}
            for k in keys:
                k = k.strip()
                try:
                    key_count[k] += 1
                except KeyError:
                    key_count[k] = 1
            clusters_df.loc[clust, "keys"] = "; ".join(["{} ({})".format(x, y) for x, y in key_count.items()])
    gene2genome.reset_index(inplace=True)
    clusters_df.reset_index(inplace=True)
    profiles_df = gene2genome.loc[:, ["contig_id", "cluster"]].drop_duplicates()
    profiles_df.columns = ["contig_id", "pc_id"]
    contigs_df = pd.DataFrame(gene2genome.fillna(0).groupby("contig_id").count().protein_id)
    contigs_df.index.name = "contig_id"
    contigs_df.columns = ["proteins"]
    contigs_df.reset_index(inplace=True)
    return gene2genome, clusters_df, profiles_df, contigs_df

print("Building the cluster and profiles (this may take some time...)")

protein_df, clusters_df, profiles_df, contigs_df = build_clusters(pcs_fp, gene2genome_df)

print("Saving files")
dfs = [gene2genome_df, contigs_df, clusters_df]
names = ['proteins', 'contigs', 'pcs']
output_dir = out_f



for name, df in zip(names, dfs):
    fn = "Cyber_{}.csv".format(name)
    fp = os.path.join(output_dir, fn)
    index_id = name.strip('s') + '_id'
    if not os.path.exists(fp):
        df.set_index(index_id).to_csv(fp)
    else:
        print("File {} exists and will be used. Use -f to overwrite.".format(fn))

        
        
        
profiles_fn = "Cyber_profiles.csv"
profiles_fp = os.path.join(out_f, profiles_fn)
if not os.path.exists(profiles_fp):
    profiles_df.to_csv(profiles_fp, index=False)
else:
    print("File {} exists and will be used. Use -f to overwrite.".format(profiles_fn))



# Create P-edges
def build_pc_matrices(profiles, contigs, pcs):
    """
    Build the pc profiles matrices (shared & singletons) from dataframes.

    Args:
        profiles (dataframe): required fields are contig_id and pc_id. # pos, contig_id, pc_id
        contigs (dataframe): contigs info, required field are proteins, pos and id. # pos, contig_id, proteins
        pcs (dataframe): pcs info, required field are pos and id.  # pos, id, size, annotated

    Returns:
        (tuple of sparse matrix): Shared PCs and singletons matrix.
    """
    pc_by_cont = profiles.groupby("contig_id").count().pc_id
    pc_by_cont = pd.merge(contigs.sort_values("pos").loc[:, ["pos", "contig_id", "proteins"]], pc_by_cont.to_frame(), how="left",
                          left_on="contig_id", right_on="contig_id").fillna(0)
    singletons = (pc_by_cont.proteins - pc_by_cont.pc_id).values
    singletons = sparse.lil_matrix(singletons).transpose()
    # Matrix
    profiles.index.name = "pos"
    profiles.reset_index(inplace=True)
    # pc_id or contig?
    profiles = pd.merge(profiles, pcs.loc[:, ["pc_id", "pos"]], left_on="pc_id", right_on="pc_id", how="inner",
                            suffixes=["", "_pc"])  # pos, contig_id, pc_id, id (pc), pos_pc
    profiles = pd.merge(profiles, contigs.loc[:, ["contig_id", "pos"]], left_on="contig_id", right_on="contig_id", how="inner",
                            suffixes=["", "_contig"])
    profiles = profiles.loc[:, ["pos_contig", "pos_pc"]]
    matrix = sparse.coo_matrix(([1]*len(profiles), (zip(*profiles.values))), shape=(len(contigs), len(pcs)),
                               dtype="bool")
    return matrix.tocsr(), singletons.tocsr()

def create_network(matrix, singletons, thres=1, max_sig=1000):
    """
    Compute the hypergeometric-similarity contig network.

    Args:
        matrix (scipy.sparse)x: contigs x protein clusters :
            M(c,p) == True <-> PC p is in Contig c.
        thres (float): Minimal significativity to store an edge value.
        max_sig (int): Maximum significance score
        
    Return
        scipy.sparse: S symmetric lil matrix, contigs x contigs.
        S(c,c) = sig(link)
    """
    contigs, pcs = matrix.shape
    pcs += singletons.sum()
    # Number of comparisons
    T = 0.5 * contigs * (contigs - 1)
    logT = np.log10(T)
    # Number of protein clusters in each contig
    # = # shared pcs + #singletons
    number_of_pc = matrix.sum(1) + singletons
    number_of_pc = number_of_pc.A1  # Transform into a flat array
    # Number of common protein clusters between two contigs, tuple + commons
    commons_pc = matrix.dot(sparse.csr_matrix(matrix.transpose(), dtype=int))
    S = sparse.lil_matrix((contigs, contigs))
    total_c = float(commons_pc.getnnz())
    i = 0  # Display
    for A, B in zip(*commons_pc.nonzero()):  # For A & B sharing contigs
        if A != B:
            # choose(a, k) * choose(C - a, b - k) / choose(C, b)
            # sf(k) = survival function = 1 -cdf(k) = 1 - P(x<k) = P(x>k)
            # sf(k-1)= P(x>k-1) = P(x>=k)
            # It is symmetric but I put the smallest before to avoid numerical bias.
            a, b = sorted([number_of_pc[A], number_of_pc[B]])
            pval = stats.hypergeom.sf(commons_pc[A, B] - 1, pcs, a, b)
            sig = min(max_sig, np.nan_to_num(-np.log10(pval) - logT))
            if sig > thres:
                S[min(A, B), max(A, B)] = sig
            # Display
            i += 1
            if i % 1000 == 0:
                sys.stdout.write(".")
            if i % 10000 == 0:
                sys.stdout.write("{:6.2%} {}/{}\n".format(i / total_c, i, total_c))
    S += S.T  # Symmetry
    S = S.tocsr()
    if len(S.data) != 0:
        print("Hypergeometric contig-similarity network:\n {0:10} contigs,\n {1:10} edges (min:{2:.2}"
                    "max: {3:.2}, threshold was {4})".format(contigs, S.getnnz(), S.data.min(), S.data.max(), thres))
    else:
        raise ValueError("No edge in the similarity network !") 
    return S


# Loding dataset
contigs_df = pd.read_csv("out/Cyber_contigs.csv")
clusters_df = pd.read_csv("out/Cyber_pcs.csv")
profiles_df = pd.read_csv("out/Cyber_profiles.csv")

# Replace names
contigs_csv_df = contigs_df.copy()
contigs_csv_df['contig_id'] = contigs_csv_df['contig_id'].str.replace(' ', '~')
print("Read {} entries from {}".format(len(contigs_csv_df), os.path.join(output_dir, '{}_contigs.csv'.format(name))))
contigs_csv_df.index.name = "pos"
contigs_csv_df.reset_index(inplace=True)

pcs_csv_df = clusters_df.copy()
profiles = profiles_df.copy()
profiles['contig_id'] = profiles['contig_id'].str.replace(' ', '~')  # ClusterONE can't handle spaces

# Filtering the PC profiles that appears only once
before_filter = len(profiles)
cont_by_pc = profiles.groupby("pc_id").count().contig_id.reset_index()

# get the number of contigs for each pcs and add it to the dataframe
cont_by_pc.columns = ["pc_id", "nb_proteins"]
pcs_csv_df = pd.merge(pcs_csv_df, cont_by_pc, left_on="pc_id", right_on="pc_id", how="left")
pcs_csv_df.fillna({"nb_proteins": 0}, inplace=True)

# Drop the pcs that <= 1 contig from the profiles.
pcs_csv_df = pcs_csv_df[pcs_csv_df['nb_proteins'] > 1]  # .query("nb_contigs>1")
at_least_a_cont = cont_by_pc[cont_by_pc['nb_proteins'] > 1]  # cont_by_pc.query("nb_contigs>1")
profiles = profiles[profiles['pc_id'].isin(at_least_a_cont.pc_id)]
print("Read {} entries (dropped {} singletons) from {}".format(len(profiles), (before_filter - len(profiles)), profiles_fp))
pcs_csv_df = pcs_csv_df.reset_index(drop=True)
pcs_csv_df.index.name = "pos"
pcs_csv_df = pcs_csv_df.reset_index()

matrix, singletons = build_pc_matrices(profiles, contigs_csv_df, pcs_csv_df)
profiles_csv = {"matrix": matrix, "singletons": singletons}
merged_df = contigs_csv_df
merged_fp = os.path.join(output_dir, 'merged_df.csv')
merged_df.to_csv(merged_fp)

ntw = create_network(matrix, singletons, thres=1, max_sig=300)


def to_clusterer(matrix, fi, contigs=None,names=None):
    """Save a network in a file ready for MCL and/or ClusterONE

    Args:
        matrix (scipy.sparse_matrix): network.
        fi (str): filename .
        names (pandas.dataframe): with the columns
            "pos":  (int) is the position in the matrix.
            "id": (str) column contain the id of the node.
            If None, self.contigs is used.

    Returns:
        str: filename
    """
    names = contigs if names is None else names
    names = names.set_index("pos").contig_id
    with open(fi, "wt") as f:
        matrix = sparse.dok_matrix(matrix)
        for r, c in zip(*matrix.nonzero()):
            f.write(" ".join([str(x) for x in (names[r], names[c], matrix[r, c])]))
            f.write("\n")
    print("Saving network in file {0} ({1} lines).".format(fi, matrix.getnnz()))
    return fi

fi = to_clusterer(ntw, out_f+"network.ntw", merged_df.copy())


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
class_to_label = {80:80,75:75,85:85,3:3,88:88,52:52,46:46,53:53,82:82,97:97,55:55,60:60,7:7,98:98,21:21,89:89,6:6,44:44,66:66,81:81,18:18,48:48,12:12,83:83,31:31,23:23,10:10,99:99,65:65,100:100,106:106,102:102,94:94,105:105,54:54,15:15,16:16,33:33,19:19,5:5,45:45,72:72,61:61,64:64,2:2,108:108,107:107,73:73,13:13,58:58,39:39,70:70,32:32,27:27,37:37,103:103,43:43,36:36,56:56,49:49,86:86,42:42,20:20,92:92,90:90,22:22,74:74,63:63,71:71,24:24,9:9,28:28,69:69,29:29,93:93,26:26,50:50,76:76,95:95,47:47,87:87,67:67,104:104,25:25,78:78,35:35,11:11,101:101,8:8,84:84,68:68,96:96,4:4,0:0,41:41,77:77,38:38,40:40,14:14,79:79,51:51,30:30,62:62,57:57,91:91,34:34,59:59,17:17,1:1}
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
            print(node,"error")

feature = np.array(feature)
if mode == "testing":
    pkl.dump(feature, open("Cyber_data/contig.feature", "wb" ) )
else:
    pkl.dump(feature, open("Cyber_data/contig.feature", "wb" ) )


# Graph check for each testing samples
cnt = 0
for node in G.nodes:
    flag = 0
    if "~" not in node:
        neighbor_label = []
        for edge in G.edges(node):
            neighbor = edge[1]
            if "~" in neighbor:
                neighbor_label.append(class_to_label[contig_to_family[neighbor]])
                flag =  1
        if flag == 0:
            label[test_to_id[node]] = -2
            #print(label[test_to_id[node]])
        if len(set(neighbor_label)) == 1:
            label[test_to_id[node]] = neighbor_label[0]
            cnt += 1



with open("network/phage_"+str(args.n)+".ntw","w") as out_f:
    for node in G.nodes:
        for edge in G.edges(node):
            neighbor = edge[1]
            out_f.write(node+","+neighbor+"\n")

pkl.dump(label, open("Cyber_data/contig.label", "wb" ) )
