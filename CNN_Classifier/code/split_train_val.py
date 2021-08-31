import random
import os
def split_train_val(family, order):
    species_list = os.listdir(order_path+order+'/'+family)
    num = len(species_list)
    random.shuffle(species_list)
    train_set = species_list[:-1]
    val_set = species_list[-1:]

    train = " ".join(train_set)
    val = " ".join(val_set)
    try:
        os.system('cd '+order_path+order+'/'+family+' && mkdir train && cp '+train+' train/')
        os.system('cd '+order_path+order+'/'+family+' && mkdir val && cp '+val+' val/')
        os.system('cd '+order_path+order+'/'+family+' && cat train/* >../'+family+'_train.fasta')
        os.system('cd '+order_path+order+'/'+family+' && cat val/* > ../'+family+'_val.fasta')
        os.system('cd '+order_path+order+' && cat *_train.fasta > ../'+order+'_train.fasta')
        os.system('cd '+order_path+order+' && cat *_val.fasta > ../'+order+'_val.fasta')
    except:
        pass

    print(family+" Done")




order_path = "/home/jyshang2/Endless_October/validation/data/"
order_list = os.listdir(order_path)

for order in order_list:
    family_list = os.listdir(order_path+order)
    for family in family_list:
        split_train_val(family, order)
