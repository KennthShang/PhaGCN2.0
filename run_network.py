import os
import subprocess
def file_number(path):
    current_path = os.getcwd()
    folder_path = current_path+"/"+path
    list = os.listdir(folder_path)
    list = [f for f in list if os.path.isfile(os.path.join(folder_path, f))]
    count = len(list)
    return count
def index_replace(n): #索引替换
    
    data_file = "network/phage_" + str(n) + ".ntw"
    index_file = "pred/contig_" + str(n) + ".csv"
    medium = "middle_" + str(n)+ ".ntw"
    # 读取索引文件
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
if file_number("network") == file_number("pred"):
    for i in range(0,file_number("network")):
        index_replace(i)
    cmd1 = "cat middle* > medium.txt"
    out = subprocess.check_call(cmd1, shell=True)
    cmd2 = "rm -rf middle*"
    out = subprocess.check_call(cmd2, shell=True)
    with open("medium.txt", "r") as f:
        lines = f.readlines()
    unique_lines = set(lines)
    with open("final_network.ntw", "w") as f:
        print("node1,node2")
        f.writelines(unique_lines)
    cmd3 = "rm -rf medium.txt"
    out = subprocess.check_call(cmd3, shell=True)