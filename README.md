# PhaGCN

PhaGCN is a GCN based model, which can learn the species masking feature via deep learning classifier, for new Phage taxonomy classification. To use PhaGCN, you only need to input your contigs to the program.


# Required Dependencies
* Python 3.x
* Numpy
* Pytorch
* Networkx
* Pandas
* [Diamond](https://github.com/bbuchfink/diamond)

All these packages can be installed using Anaconda.

If you want to use the gpu to accelerate the program:
* cuda 10.1 
* Pytorch-gpu

## An easiler way to install
We recommend you to install all the package with [Anaconda](https://anaconda.org/)

After cloning this respository, you can use anaconda to install the **environment.yaml**. This will install all packages you need with gpu mode (make sure you have installed cuda on your system).

# Usage (example)
Here we present an example to show how to run PhaGCN. We support a file named "contigs.fa" in the Github folder and it contain contigs simulated from E. coli phage. The only command that you need to run is `python run_Speed_up.py --contigs contigs.fa -len 8000`. 

There are two parameters for the program: 1. `--contigs` is the path of your contigs file. 2. `--len` is the length of the contigs you want to predict. As shown in our paper, with the length of contigs increases, the recall and precision increase. We recommend you to choose a proper length according to your needs. The default length is 8000bp. 

The output file is **final_prediction.csv**. There are three column in this csv file: "contig_name, median_file_name, prediction".

The given database only support prediction under the **Caudovirales** order. But you can change the database if required.

# Notice
If you want to use PhaGCN, you need to take care of two things:
1. Make sure all your contigs are virus contigs. You can sperate bacteria contigs by using [VirSorter](https://github.com/simroux/VirSorter) or [DeepVirFinder](https://github.com/jessieren/DeepVirFinder)
2. The script will pass contigs with non-ACGT characters, which means those non-ACGT contigs will be remained unpredict.


# References
how to cite this tool:
```
Jiayu Shang, Jingzhe Jiang and Yanni Sun, Bacteriophage classification for assembled contigs using Graph Convolutional Network, submitted to ISMB 2021

Or

Shang, J., Jiang, J., & Sun, Y. (2021). Bacteriophage classification for assembled contigs using Graph Convolutional Network. arXiv preprint arXiv:2102.03746.
```

## Supplementary information
The supplementary file of the paper can be found in the supplementary folder.

## Contact
If you have any questions, please email us: jyshang2-c@my.cityu.edu.hk
