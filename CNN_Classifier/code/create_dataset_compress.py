import os
import numpy as np






if __name__ == '__main__':
    Load_path = "int_val/"
    name_list = os.listdir(Load_path)
    name_list.sort()

    #cnt = 0
    data = []

    for name in name_list:
        read = np.genfromtxt(Load_path+name, delimiter=',')
        if read.reshape(-1).shape[0] == 1698:
            read = read.reshape(1, 1698)
        
        cnt = int(name.split(".")[0])
        label = np.zeros(len(read)) + cnt
        
        read = np.c_[read, label]


        data.append(read)
        #cnt += 1

        #print(name + " finished !")
        np.savetxt("dataset/"+name, read, delimiter=",", fmt='%d')
    #data = np.concatenate(data, axis=0)
     
    #np.savetxt("dataset/val.csv", data, delimiter=",", fmt='%d')
