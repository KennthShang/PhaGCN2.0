import os
import numpy as np






if __name__ == '__main__':
    Load_path = "int_train/"
    name_list = os.listdir(Load_path)
    name_list.sort()
    cnt = 0
    data= []

    for name in name_list:
        read = np.genfromtxt(Load_path+name, delimiter=',')
        np.random.shuffle(read)


        label = np.zeros(len(read)) + cnt
        read = np.c_[read, label]



        data.append(read)

        cnt += 1

        #print(name + " finished !")

    data = np.concatenate(data, axis=0)
    
    np.savetxt("dataset/train.csv", data, delimiter=",", fmt='%d')



