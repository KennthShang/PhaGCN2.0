import numpy as np
import pandas as pd
import os
import Bio
from Bio import SeqIO
import subprocess
import argparse
import re
if not os.path.exists("input"):
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

def special_match(strg, search=re.compile(r'[^ACGT]').search):
    return not bool(search(strg))
cnt = 0
file_id = 0
records = []
for record in SeqIO.parse("all_simple_pre.fasta", 'fasta'):
    if cnt !=0 and cnt%1000 == 0:
        SeqIO.write(records, "input/contig_"+str(file_id)+".fasta","fasta")
        records = []
        file_id+=1
        cnt = 0
    seq = str(record.seq)
    seq = seq.upper()
    if special_match(seq):
        if len(record.seq) > 1700:
            records.append(record)
            cnt+=1

SeqIO.write(records, "input/contig_"+str(file_id)+".fasta","fasta")
file_id+=1
cmd1 = "grep \">\" input/*.fasta > result.txt"
out = subprocess.check_call(cmd1, shell=True)


f = open("result.txt","r")    #建立reference_name_id.csv文件
g = open("reference_name_id.csv","w")
n = 0
print("name,idx",file=g)
for each in f:
    each = each.replace("\n","")
    (a,b) = each.split(":",1)
    (b,c) = b.split(" ",1)

    print(c+","+str(n),file=g)
    n = n + 1
f.close()
g.close()
cmd2 = "grep \">\" all_simple_pre.fasta > result1.txt"
out = subprocess.check_call(cmd2, shell=True)


f = open("result.txt","r")  #建立ALL_protein.fasta
records = []

for each in f :
    each = each.replace("\n", "")
    (a, b) = each.split(":", 1)
    (b, c) = b.split(" ", 1)
    b = b.replace(">","")
    for record in SeqIO.parse("all_simple_pre.fasta", 'fasta'):
        if record.id == b:
            records.append(record)
        else:
            continue
SeqIO.write(records, "ALL.fasta", "fasta")
f.close()

cmd3 = "prodigal -i ALL.fasta -a 0014_protein.fasta -f gff -p meta"
out = subprocess.check_call(cmd3, shell=True)
f = open("result.txt","r")
g = open("protein.py","w")
print("f = open(\"0014_protein.fasta\",\"r\")",file=g)
print("g = open(\"ALL_protein.fasta\",\"w\")",file=g)
print("for each in f:",file=g)
print("    each = each.replace(\"\\n\",\"\")",file=g)
print("    if \">\" in each:",file=g)
print("        (x,y) = each.split(\" \",1)",file=g)
print("        c = x",file=g)
print("        (a, b) = x.rsplit(\"_\", 1)",file=g)
print("        x = x.replace(\"_\",\"\")",file=g)
print("        x = x.replace(\".\",\"\")",file=g)
print("        x = x.replace(\">\", \"\")",file=g)
for each in f:
    each = each.replace("\n","")
    (a,b) = each.split(":",1)
    (b,c) = b.split(" ",1)
    print("        a = a.replace(\""+b+"\",\""+c+"\")",file=g)
print("        a = a.replace(\"]\",\"\")",file=g)
print("        a = a.replace(\"[\",\"\")",file=g)
print("        if \">\" in a:",file=g)
print("            print(a)",file=g)
print("        each = c+\" |\"+x+\" [\"+a+\"]\"",file=g)
print("    else:",file=g)
print("        each = each",file=g)
print("    print(each,file=g)",file=g)
g.close()
f.close()
cmd4 = "python3 protein.py"
out = subprocess.check_call(cmd4, shell=True)
f = open("ALL_protein.fasta","r")    #建立ALL_gene_to_genomes.csv，和ALL_genome_profile.csv
g1 = open("ALL_gene_to_genomes.csv","w")
print("protein_id,contig_id,keywords",file=g1)
g2 = open("ALL_genome_profile.csv","w")
print("contig_id,proteins",file=g2)
m = []
for each in f:
    if ">" in each:
        each = each.replace("\n","")
        (x,y) = each.split(" |",1)
        x = x.replace(">","")
        (y,z) = y.split("[",1)
        z = z.replace("]","")
        y = y.replace(" ","")
        print(x+","+z+","+y,file=g1)
        m.append(z)
x = set(m)
for i in x:
    b = m.count(i)
    i = i.replace("\'","")
    i = i.replace(" ", "~")
    print(i+","+str(b),file = g2)
f.close()
g1.close()
g2.close()

f1 = open("taxa.txt","r")    #建立taxonomic_label.csv及部分代码行修改部分
g1 = open("taxonomic_label.txt","w")
def pipei(x,y):
    f = open("result1.txt", "r")
    for eachline in f:
        eachline = eachline.replace("\n", "")
        (a, b) = eachline.split(">", 1)
        (b, c) = b.split(".", 1)
        (c, d) = c.split(" ", 1)
        b = b.replace(">", "")
        if b in x:
            return d+","+y
for eachline2 in f1:
    eachline2 = eachline2.replace("\n","")
    (x,y) = eachline2.split("\t",1)
    if pipei(x,y) == None:
        print(x,y)
    else:
        print(pipei(x,y),file=g1)
f1.close()
g1.close()
f1 = open("taxonomic_label.txt","r")
g1 = open("daimahang1.txt","w")
g3 = open("daimahang2.txt","w")
pd.set_option('display.max_columns', None)   #设置输出列数不会被省略
pd.set_option('display.max_rows', None)   #设置输出行数不会被省略
pd.set_option('display.width',None)     #设置一次输出多少行
pd.set_option('max_colwidth',None)
x = ("contig_id","b","class","family")
count=0
for line in f1.readlines():
    count=count+1
df = pd.DataFrame(columns =x ,index = range(1,count+1) )
count1 = 0
f1.close()
f1 = open("taxonomic_label.txt","r")
for eachline in f1:
    eachline = eachline.replace("\n","")
    (x,y) = eachline.split(",",1)
    x = x.replace(" ","~")
    count1 = count1+1
    df.loc[count1]["contig_id"] = x
    df.loc[count1]["b"] = y
i = 1
df.loc[1]["class"] = 0
df.loc[1]["family"] = 0
while i < count:
    if df.loc[i]["b"] != df.loc[i+1]["b"]:
        df.loc[i+1]["class"] = df.loc[i]["class"]+1
        df.loc[i + 1]["family"] = df.loc[i]["family"] + 1
    else:
        df.loc[i+1]["class"] = df.loc[i]["class"]
        df.loc[i + 1]["family"] = df.loc[i]["family"]
    i = i + 1
df1 = df.drop(columns=["b"])
df2 = df.drop(columns=["contig_id","class"])
df3 = df.drop(columns=["contig_id","b"])
df1.to_csv("taxonomic_label.csv",sep=",",index=False,header=True)
df2.to_csv("daimahang1.txt",sep=":",index=False,header=False)
df3.to_csv("daimahang2.txt",sep=":",index=False,header=False)
f1.close()
def quchongfu(f,g):
    a = []
    for eachline in f:
        eachline = eachline.replace("\n", "")
        a.append(eachline)
    x = set(a)
    for i in x:
        i = i.replace("\'", "")
        (x,y) = i.split(":",1)
        i = y + ":\"" + x +"\""
        print(i+",", file=g)
f1 = open("daimahang1.txt","r")
f2 = open("daimahang2.txt","r")
g1 = open("jieguo1.txt","w")
g2 = open("jieguo2.txt","w")
quchongfu(f1,g1)
quchongfu(f2,g2)
f1.close()
f2.close()
g1.close()
g2.close()
f1 = open("jieguo1.txt","r")
f2 = open("jieguo2.txt","r")
g1 = open("code.txt","w")
def duohanghebing1(f,g):
    a = ''  # 空字符（中间不加空格）
    for line in f:
        a += line.strip()  # strip()是去掉每行末尾的换行符\n 1
    c = a.split()  # 将a分割成每个字符串 2
    b = ''.join(c)  # 将c的每个字符不以任何符号直接连接 3
    b = b.strip(",")
    print("pred_to_label = {"+b+"}",file=g)
def duohanghebing2(f,g):
    a = ''  # 空字符（中间不加空格）
    for line in f:
        a += line.strip()  # strip()是去掉每行末尾的换行符\n 1
    c = a.split()  # 将a分割成每个字符串 2
    b = ''.join(c)  # 将c的每个字符不以任何符号直接连接 3
    b = b.strip(",")
    b = b.replace("\"","")
    print("class_to_label = {"+b+"}",file=g)
duohanghebing1(f1,g1)
duohanghebing2(f2,g1)
if not os.path.exists("result"):
    _ = os.makedirs("result")
else:
    print("folder {0} exist... cleaning dictionary".format("result"))
    if os.listdir("result"):
        try:
            _ = subprocess.check_call("rm -rf {0}".format("result"), shell=True)
            _ = os.makedirs("result")
            print("Dictionary cleaned")
        except:
            print("Cannot clean your folder... permission denied")
            exit(1)
cmd5 = "sh clean_all_text.sh"
out = subprocess.check_call(cmd5, shell=True)