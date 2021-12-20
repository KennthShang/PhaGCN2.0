import sys
import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

def create_reads(file_name):
    new_record = []
    for record in SeqIO.parse("validation/"+file_name, "fasta"):
        seq = record.seq
        seq = seq.upper()
        if len(seq) > 1700:
            for i in range(0, len(seq), 1700):
                if i + 1700 > len(seq):
                    new_seq = seq[-1700:]
                    rec = SeqRecord(new_seq, id=record.id, description=record.description, name=record.name)
                    new_record.append(rec)
                    break
                
                new_seq = seq[i:i+1700]
                rec = SeqRecord(new_seq, id=record.id, description=record.description, name=record.name)
                new_record.append(rec)
        else:
            print("error length < 2000bp")
            print(record.description)
            
            
    SeqIO.write(new_record, "split_long_reads_val/"+file_name, "fasta")
    
if __name__ == "__main__":
    path = "validation/"
    name_list = os.listdir(path)
    for name in name_list:
        create_reads(name)
        #print(name + " finished")    

