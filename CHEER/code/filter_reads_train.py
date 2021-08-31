import numpy as np
import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

def filter_reads(pos, file_name):
    with open("filtered_train/"+file_name.split(".")[0]+"_new.fasta", 'w') as f_out:
        for record in SeqIO.parse(pos+file_name, "fasta"):
            read = str(record.seq)
            flag = 0
            for nucl in read:
                if nucl == 'A':
                    continue
                elif nucl == 'C':
                    continue
                elif nucl == 'G':
                    continue
                elif nucl == 'T':
                    continue
                else:
                    flag = 1

                    break
            if flag == 0:
                SeqIO.write(record, f_out, "fasta")
                


if __name__ == "__main__":
    load_path = "split_long_reads_train/"
    
    name_list = os.listdir(load_path)
    for name in name_list:
        filter_reads(load_path, name)
       
