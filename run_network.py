import os
import subprocess
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--outpath', type=str, default = "result")
args = parser.parse_args()
def file_number(path):
    current_path = os.getcwd()
    folder_path = current_path+"/"+path
    list = os.listdir(folder_path)
    list = [f for f in list if os.path.isfile(os.path.join(folder_path, f))]
    count = len(list)
    return count
def index_replace(n): 
    
    data_file = f"{args.outpath}/network/phage_" + str(n) + ".ntw"
    index_file = f"{args.outpath}/pred/contig_" + str(n) + ".csv"
    medium = f"{args.outpath}/middle_" + str(n)+ ".ntw"
    index_dict = {}
    with open(index_file, 'r') as f:
        for line in f:
            fields = line.strip().split(",")
            k = fields[0]
            v = fields[1]
            index_dict[v] = "test_"+str(n)+"_"+k

    # 替换数据文件内容
    with open(data_file, 'r') as f1:
        lines = f1.readlines()
    with open(medium, 'w') as f2:
        for line in lines:
            fields = line.strip().split(",")
            for i in range(len(fields)):
                if fields[i] in index_dict:
                    fields[i] = index_dict[fields[i]]
            f2.write(','.join(fields) + '\n')

print("start combine network........")
if file_number(f"{args.outpath}/network") == file_number(f"{args.outpath}/pred"):
    for i in range(0,file_number(f"{args.outpath}/network")):
        index_replace(i)
    cmd1 = f"cat {args.outpath}/middle* > medium.txt"
    out = subprocess.check_call(cmd1, shell=True)
    cmd2 = f"rm -rf {args.outpath}/middle*"
    out = subprocess.check_call(cmd2, shell=True)
    with open("medium.txt", "r") as f:
        lines = f.readlines()
    unique_lines = set(lines)
    with open(f"{args.outpath}/final_network.ntw", "w") as f:
        f.writelines("node1,node2")
        f.writelines(unique_lines)
    cmd3 = "rm -rf medium.txt"
    out = subprocess.check_call(cmd3, shell=True)

import pandas as pd
def create_dict_from_first_file(file1):
    df1 = pd.read_csv(file1, sep='\t', header=None)
    result_dict = {}
    for index, row in df1.iterrows():
        key = row[0] 
        values = row[1]
        result_dict[key] = values
    return result_dict
f1 = open(f"{args.outpath}/result.txt","r")
result_dict = create_dict_from_first_file("database/taxonomic_path.csv")
g1 = open(f"{args.outpath}/final_prediction.csv","w")
print("contig_name,idx,prediction,full_path",file = g1)
for each in f1:
    each = each.strip()
    (x,y,z) = each.split(",",2)
    if "idx" == y:
        continue
    else:
        if "_like" in z:
            index,_ = z.split("_",1)
            taxa_path = f"{result_dict[index]}_like"
        else:
            index = z
            taxa_path = result_dict[index]
        print(f"{each},{taxa_path}",file = g1)
cmd3 = f"rm -rf {args.outpath}/result.txt"
out = subprocess.check_call(cmd3, shell=True)
