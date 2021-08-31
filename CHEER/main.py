import utils
import numpy as np 
import os

# Rejection layer
#print("Running rejection layer !!!")
os.system("bash code/pre_train_script.sh")
os.system("python show_result.py --gpus 0 --n 2 --t 0.6 --embed ../pkl/rejection/embed.pkl --classifier ../pkl/rejection/Reject_params.pkl --rejection Y")
os.system("python split_data.py --path non_rna --dir rna non_rna --child_list rna non_rna")
tmp = os.listdir("prediction")
if "early_stop.fasta" in tmp:
    os.system("mv prediction/early_stop.fasta prediction/non_rna/")
print("Rejection layer finished!!!")


# Order layer
print("Running Order Classifier !!!")
os.system("bash clean_all_pre_script.sh")
os.system("cp prediction/rna/* validation")
os.system("bash code/pre_train_script.sh")
os.system("python show_result.py --gpus 0 --n 6 --t 0.6 --embed ../pkl/Order/embed.pkl --classifier ../pkl/Order/order_params.pkl")
order_list = list(utils.order_dict.values())
order_name_string = " ".join(order_list)

order_dir_list = ["rna/"+name for name in order_list]
order_dir_string = " ".join(order_dir_list)
os.system("python split_data.py --path rna --dir "+ order_dir_string + " --child_list "+ order_name_string)
tmp = os.listdir("prediction")
if "early_stop.fasta" in tmp:
    os.system("mv prediction/early_stop.fasta prediction/rna/")
print("Order Classifier finished!!!")

# Family layer
order_folder = os.listdir("prediction/rna/")
path = "prediction/rna/"
for order in order_folder:
    if len(order.split(".")) == 2:
        continue
    cnt = utils.get_leaf_num(order)
    if cnt == 0:
        continue
    # Start to run CHEER
    os.system("bash clean_all_pre_script.sh")
    os.system("cp "+path+order+"/* validation")
    os.system("bash code/pre_train_script.sh")
    os.system("python show_result.py --gpus 0 --n "+str(cnt)+" --t 0.6 --embed ../pkl/Family/"+ order +"_Embed.pkl --classifier ../pkl/Family/"+ order +"_Params.pkl")

    family_dict = utils.get_dict(order)
    family_list = list(family_dict.values())
    family_name_string = " ".join(family_list)
    
    family_dir_list = ["rna/"+order+"/"+name for name in family_list]
    family_dir_string = " ".join(family_dir_list)

    os.system("python split_data.py --path rna/"+order+" --dir "+ family_dir_string + " --child_list "+ family_name_string)
    tmp = os.listdir("prediction")
    if "early_stop.fasta" in tmp:
        os.system("mv prediction/early_stop.fasta prediction/rna/"+order+"/")

    # Genus layer
    family_folder = os.listdir("prediction/rna/"+order+"/")
    for family in family_folder:
        if len(order.split(".")) == 2:
            continue
        cnt = utils.get_leaf_num(family)
        if cnt == 0:
            continue
        # Start to run CHEER
        os.system("bash clean_all_pre_script.sh")
        os.system("cp "+path+order+"/" +family+"/*"+ " validation")
        os.system("bash code/pre_train_script.sh")
        os.system("python show_result.py --gpus 0 --n "+str(cnt)+" --t 0.6 --embed ../pkl/Genus/"+ family +"_Embed.pkl --classifier ../pkl/Genus/"+ family +"_Params.pkl")

        genus_dict = utils.get_dict(family)
        genus_list = list(genus_dict.values())
        genus_name_string = " ".join(genus_list)

        genus_dir_list = ["rna/"+order+"/"+family+"/"+name for name in genus_list]
        genus_dir_string = " ".join(genus_dir_list)
        
        os.system("python split_data.py --path rna/"+order+"/"+family+" --dir "+ genus_dir_string + " --child_list "+ genus_name_string)
        tmp = os.listdir("prediction")
        if "early_stop.fasta" in tmp:
            os.system("mv prediction/early_stop.fasta prediction/rna/"+order+"/"+family+"/")
        print(family+" is finished !!!")
    print(order+" is finished !!!")
