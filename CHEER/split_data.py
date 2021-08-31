import numpy as np 
import argparse
import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--path', type=str, default='rna')
parser.add_argument('--dir', nargs='+', help='<Required> Set flag', required=True)
parser.add_argument('--child_list', nargs='+', help='<Required> Set flag', required=True)
args = parser.parse_args()


name_list = args.child_list
dir_list = args.dir
path = os.getcwd()
new_name = os.listdir(path+"/filtered_val/")[0]
new_fasta = list(SeqIO.parse(path+"/filtered_val/"+new_name, "fasta"))
ori_name = os.listdir(path+"/validation/")[0]
ori_fasta = list(SeqIO.parse(path+"/validation/"+ori_name, "fasta"))

os.chdir("prediction")


fasta_id = {}
id_to_pred = {}
num_class = set()

# record prediction for each 250bp reads
with open("result.txt") as file_in:
    for line in file_in.readlines():
        tmp = line.replace('\n', '').split("->")
        id_ = int(tmp[0])
        class_ =int(tmp[1])
        # record prediction
        try:
            if new_fasta[id_-1].id not in fasta_id.keys():
                fasta_id[new_fasta[id_-1].id] = []
            fasta_id[new_fasta[id_-1].id].append(class_)       
            num_class.add(class_)
        except:
            print("Index error")
            print("index = ", str(id_))
            print("length of fasta", str(len(new_fasta)))
            exit(1)


# combine short reads prediction for original reads
too_short_reads = []
for record in ori_fasta:
    key = record.id
    if key not in fasta_id.keys():
        too_short_reads.append(key)
        continue
    nums = fasta_id[key]
    counts = np.bincount(nums)
    id_to_pred[key] = np.argmax(counts)
    
    

    
# write the early stop reads
flag = 0
with open("error_reads.fasta", "w") as short_handle:
    with open("early_stop.fasta", "w") as output_handle:
        for record in ori_fasta:
            if record.id in too_short_reads:
                record.description = "Too_short_or_error"
                SeqIO.write(record, short_handle, "fasta")
                
            elif id_to_pred[record.id] == 0:
                record.description = args.path
                SeqIO.write(record, output_handle, "fasta")
                flag = 1
    if flag == 0:
        os.system("rm early_stop.fasta")

        
# write other reads into different class folder
for i in num_class:
    flag = 0
    if i == 0:
        continue
    os.system("mkdir "+dir_list[i-1])
    with open(dir_list[i-1]+"/"+name_list[i-1]+".fasta", 'w') as file_out:
        for record in ori_fasta:
            if record.id in too_short_reads:
                continue
            if id_to_pred[record.id] == i:
                record.description = dir_list[i-1]
                SeqIO.write(record, file_out, "fasta")
                flag = 1
    if flag == 0:
        os.system("rm -rf "+dir_list[i-1])

            
