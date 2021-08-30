import numpy as np
import pandas as pd
import os
import Bio
from Bio import SeqIO
import pandas as pd
import subprocess
import argparse
import re

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--contigs', type=str, default = 'contigs.fa')
parser.add_argument('--len', type=int, default=8000)
args = parser.parse_args()   #俩个命令行参数，分别为args.contigs和args.len

if not os.path.exists("input"):    #如果input文件夹不存在，就创建一个input文件夹，如果存在input文件夹，将其中内容清空
    _ = os.makedirs("input")
else:
    print("folder {0} exist... cleaning dictionary".format("input"))
    if os.listdir("input"):
        try:
            _ = subprocess.check_call("rm -rf {0}".format("input"), shell=True)
            _ = os.makedirs("input")
            print("Dictionary cleaned")
        except:
            print("Cannot clean your folder... permission denied")
            exit(1)


if not os.path.exists("pred"):   #如果pred文件夹不存在，就创建一个pred文件夹，如果存在pred文件夹，将其中内容清空
    _ = os.makedirs("pred")
else:
    print("folder {0} exist... cleaning dictionary".format("pred"))
    if os.listdir("pred"):
        try:
            _ = subprocess.check_call("rm -rf {0}".format("pred"), shell=True)
            _ = os.makedirs("pred")
            print("Dictionary cleaned")
        except:
            print("Cannot clean your folder... permission denied")
            exit(1)



if not os.path.exists("Split_files"):   #如果Split_files文件夹不存在，就创建一个Split_files文件夹，如果存在Split_files文件夹，将其中内容清空
    _ = os.makedirs("Split_files")
else:
    print("folder {0} exist... cleaning dictionary".format("Split_files"))
    if os.listdir("Split_files"):
        try:
            _ = subprocess.check_call("rm -rf {0}".format("Split_files"), shell=True)
            _ = os.makedirs("Split_files")
        except:
            print("Cannot clean your folder... permission denied")
            exit(1)



#####################################################################
##########################    Start Program  ########################
#####################################################################

def special_match(strg, search=re.compile(r'[^ACGT]').search):
    return not bool(search(strg))


cnt = 0
file_id = 0
records = []
for record in SeqIO.parse(args.contigs, 'fasta'):
# 此时record的格式为
# ID: 6_13
# Name: 6_13
# Description: 6_13 k141_1331 flag=0 multi=33.8007 len=8757
# Number of features: 0
# Seq('CACAGCTGCGACTGGGACCGGCAGACATTCTGGAGTCAGATGAGAATGGCATTA...CCG')
    if cnt !=0 and cnt%1000 == 0:
        SeqIO.write(records, "Split_files/contig_"+str(file_id)+".fasta","fasta")
        records = []     #将得到的序列分开并存储到split文件夹中，命名格式为contig_1.fasta,contig_2.fasta
        file_id+=1
        cnt = 0
    seq = str(record.seq)
    seq = seq.upper()  #小写转大写

    if special_match(seq):
        if len(record.seq) > args.len:
            records.append(record)
            cnt+=1

SeqIO.write(records, "Split_files/contig_"+str(file_id)+".fasta","fasta")
file_id+=1    #分开最后一个
for i in range(file_id):
    cmd = "mv Split_files/contig_"+str(i)+".fasta input/"
    try:
        out = subprocess.check_call(cmd, shell=True)
    except:
        print("Moving file Error for file {0}".format("contig_"+str(i)))
        continue

    cmd = "python run_CNN.py"
    try:
        out = subprocess.check_call(cmd, shell=True)
    except:
        print("Pre-trained CNN Error for file {0}".format("contig_"+str(i)))
        cmd = "rm input/*"
        out = subprocess.check_call(cmd, shell=True)
        continue
    
