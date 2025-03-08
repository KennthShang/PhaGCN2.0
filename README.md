# ATTNETION!!!

- The program has been updated and moved to [PhaBOX 2](https://github.com/KennthShang/PhaBOX), which is more user-friendly. In the new version, PhaGCN is generalized to all kinds of viruses, more than just phages. Hope you will enjoy it. This folder will be no longer 


### updates (December 23,2024)
 
implemented updates: 
 
 * **Memory Optimization**: In the previous version, PhaGCN2, we used a batch size of 1000 to save memory. This resulted in a lack of connectivity between batches in the network graph. In this update, we have adjusted the batch size to 100,000, which should be sufficient for the classification of most viral genome datasets. If with sufficient system memory, you could process millions of viral sequences in one batch.
 
 * **Runtime Optimization**: Thanks to updates in [Python 3.13](https://www.python.org/downloads/release/python-3130/?featured_on=pythonbytes), most steps now utilize multithreading. We have implemented this in PhaGCN3, significantly increasing the speed of virus classification. On a machine with 64 CPUs and 256GB of RAM, classifying 60,000 viral sequences takes about 7.5 hours. Because classification requires constructing a network graph, we do not recommend processing an excessively large number of sequences (>100000) in one batch. As the number of virus increases, the time and memory taken will increases exponentially
 
 * **Precision Optimization**: During previous testing, we identified isolated connected subgraphs in the network graph predicted as "_like".  The precision of these subgraphs was poor due to uncertainties inherent in the GCN graph.  Therefore, we extracted these clusters and introduced [Genomad](https://github.com/apcamargo/genomad/) for classification.  PhaGCN3 currently only assigns a cluster ID to these nodes，but the specific classification of this cluster is now provided by [Genomad](https://github.com/apcamargo/genomad/). Viruses with the same cluster ID exhibit high similarity.
 
 * **Results Optimization**: We have introduced a confidence score for the classification results.  Results with a confidence score above 0.5 are considered high-confidence predictions.
 
 * **Visualization Optimization**:We now support direct output of network graph visualizations, as shown in the image below. We have tested and confirmed that the current version supports visualizing network graphs with fewer than 70,000 nodes. If you need more flexible visualization options, we also provide a network source file compatible with [Gephi](https://gephi.org/) , located at **tmp/node.csv,tmp/edge.csv** in the **results** folder.
 
   <img src="https://wenguang.oss-cn-hangzhou.aliyuncs.com/figure/image-20241218170530956.png" style="zoom:50%;" />

## PhaGCN2

PhaGCN2 is a GCN based model, which can learn the species masking feature via deep learning classifier, for new virus taxonomy classification. To use PhaGCN2, you only need to input your contigs to the program.



# PhaGCN2.3 newly update
Update Log (August 29, 2024)
* Our database has now been updated based on the latest [ICTV classification tables](https://ictv.global/sites/default/files/VMR/VMR_MSL39.v1_20240912.xlsx). (Special note: The new ICTV classification database is nearly 5000 more than the previous version. This caused the program to run nearly twice as slowly as before. We are trying to optimize the basic algorithm to improve the classification speed. )
* The output file path parameter has been added. `--outpath` is the output file path of your result.
*  The classification results now include the full path of the classification information (notice: not all viruses have a complete taxonomic path, all classification paths strictly match the classification criteria of ICTV. r:Realm, sr:Subrealm, k:Kingdom, sk:Subkingdom, p:Phylum, sp:Subphylum, c:Class, sc:Subclass, o:Order, so:Suborder, f:Family, sf:Subfamily ,g:Genus). 

# PhaGCN2.2 newly update
To better show the results of the network result, we can now export the total network file **final_network.ntw**. 
In order to distinguish the test points of different batches, the test node ID are renamed as "test"+ betches number +sequence ID,such as "test_0_NC_001825.1".

# PhaGCN2.1 newly update
Our database has now been updated based on the latest [ICTV classification tables](https://ictv.global/filebrowser/download/585).
In order to make it easier for users to view the sequence classification in our database, we put the specific sequence information at **database/VMR_based_on_ICTV.csv**
Due to changes to Caudoviricetes in the new version of ICTV(remove Siphoviridae,Myoviridae and Podoviridae), we have two extension of the methods:

1. If you only care about the phage taxa classification, please use the extension version of [PhaGCN_newICTV](https://github.com/KennthShang/PhaGCN_newICTV)
2. In PhaGCN2.0, we extended phage without family labels to the genus level.

According to our test, the extension version of PhaGCN still remain high performance for the new ICTV labels at family level.
For PhaGCN2.0, there maybe some misclassifications in some genus or subfamilies of Caudoviricetes(Bronfenbrennervirinae,Nclasvirinae,Benedictvirus,Fromanvirus,Kroosvirus,Triavirus,Turbidovirus,Veracruzvirus).In fact, the PhaGCN2 results suggest that they are very similar to other genus, such as Gladiatorvirus and Backyardiganvirus.




# Required Dependencies
* Python 3.x
* Numpy
* Pytorch
* Networkx
* Pandas
* [Diamond](https://github.com/bbuchfink/diamond)

All these packages can be installed using Anaconda.

If you want to use the gpu to accelerate the program:(if you want to train your own virus classification database,these packages must be install)

* cuda 10.1
* Pytorch-gpu

# An easiler way to install
We recommend you to install all the package with Anaconda.
After cloning this respository, you can use anaconda to install the **environment.yaml**. This will install all packages you need with gpu mode (make sure you have installed cuda on your system).
We recommend you to install all the package with Anaconda.The command that you need to run is 
```bash
cd PhaGCN2.0
conda env create -f environment.yaml -n phagcn2
```

You need to prepare the database before using it.
```bash
cd database
tar -zxvf ALL_protein.tar.gz
cd ..
```
and you can use it to make virus classification.


# Usage (example)
Before you use it each time, you need to run is 
```bash
conda activate phagcn2
export MKL_SERVICE_FORCE_INTEL=1
```

Here we present an example to show how to run PhaGCN2. We support a file named "contigs.fa" in the Github folder and it contain contigs simulated from E. coli phage. The only command that you need to run is 

```bash
python run_Speed_up.py --contigs contigs.fa --len 8000
```

There are two parameters for the program: 
1. `--contigs` is the path of your contigs file. 
2. `--len` is the length of the contigs you want to predict. 
As shown in our paper, with the length of contigs increases, the recall and precision increase. We recommend you to choose a proper length according to your needs. The default length is 8000bp.
The shortest length supported is 1700bp.The output file is **final_prediction.csv**. There are three column in this csv file: "contig_name, median_file_name, prediction".

# New changes:
Now,the given database can support prediction under the all viruses which is base on [ICTV 2021 year reporter](https://talk.ictvonline.org/taxonomy/vmr/m/vmr-file-repository/13175). In prediction result,we add a prediction result named "Family_like" , if your virus species prediction label is "_like", it indicates that your virus and some viruses in the virus library are the same order but different families of the relationship.
In the **Network** folder will generate a network map file, you can use this file to draw your unique and beautiful network map

# New function
Now we support that you can train your own virus classification database.
If you want train your own virus classification database, follow these steps.

First of all,you need with gpu mode (make sure you have installed cuda on your system)and  run 
```bash
pip install bio
pip install torch
sudo apt install prodigal
cd CHEER
sh creat.sh
```

1. Step one:
In this step,you need make your virus sequences (Known families)in different folder, dividing to training set and validation set.Make sure your virus sequence is longer than 1700BP.(If you have a segmented virus, combine it into a single sequence,Otherwise, it will be divided into several different kinds of viruses)
Take [ICTV 2021 year reporter](https://talk.ictvonline.org/taxonomy/vmr/m/vmr-file-repository/13175) as an example,there are 11 viral sequences in Lipothrixviridae .
Randomly divide 9 out of 10 sequences to do the training set and 1 to do the test set.Combine the nine training set sequences into a sequence file named Lipothrixviridae and place it in the train folder.Combine the one validation set sequences into a sequence file named Lipothrixviridae and place it in the validation folder.The same goes for the other family.(Ensure that the validation set is non-empty).

Preprocess your data set:
```bash
$ bash code/re_train_script.sh
```

Train your data set CNN model
```bash
$ python3 train.py --n 8 --gpus 1 --weight "1,1,1,1,1,1,1,1"
```
`--n` is the number of your families,`--weight` is the  weight coefficient，The number of numbers in weight is equal to the number of n, `--gpu` is the number of Gpus you have
This will produce two files **Embed.pkl** and **Params.pkl**,Replace the two files with the same name in the **CNN_Classifier** folder.It requires around 250GB of memory(The larger the data set, the more memory is required).

2. Step two:
Merge all sequence files into one file and name it **all_pre.fasta**,then move it into 

**CHEER** folder.Then run:
```bash
$ python3 deal_all_pre.py
```
Please modify your data set until no label error is reported(The label should contain at least one space).
Take the sequence number and corresponding family name in a TXT text and named **taxa.txt** (separated by tabs)then place it in the CHEER folder.Run:
```bash
$ python3 deal_result.py`
```
It generates a folder of result,in this folder replaces the first line of **code.txt** with line 159 of **run_GCN.py** in the body of PhaGCN, and the second line with line 643 of **run_Knowledgegraph.py**,The other five files replace each of the five files in the **database** folder.
3. Step three:
Change the default n on line 76 of **run_CNN.py** to the number of `--n` in step two
Copy **all_simple_pre.fasta** to your **PhaGCN** folder and run():
```bash
$ python3 pre_train.py --contig all_simple_pre.fasta --len 1700
```

After running it, rename **contig.F** in the Cyber_data folder to **dataset_compressF** and replace the file with the same name in the **database** folder.

(See the **CHEER/train_example** folder for an example)
# Notice
If you want to use PhaGCN, you need to take care of three things:
1. Make sure all your contigs are virus contigs. You can sperate bacteria contigs by using [VirSorter](https://github.com/simroux/VirSorter) or [DeepVirFinder](https://github.com/jessieren/DeepVirFinder)
2. The script will pass contigs with non-ACGT characters, which means those non-ACGT contigs will be remained unpredict.
3. if the program output an error (which is caused by your machine): Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.
You can type in the command `export MKL_SERVICE_FORCE_INTEL=1` before runing **run_Speed_up.py**
4. If you want train your own virus classification database,Hardware requirements can be considerable(exceeding 48 GB,and at least one GPU), depending mainly on the size and complexity of the dataset. (Relationship between memory requirements and sequences analyzed forthcoming)


# References
how to cite this tool:
```
Jiayu Shang, Jingzhe Jiang, Yanni Sun, Bacteriophage classification for assembled contigs using graph convolutional network, Bioinformatics, Volume 37, Issue Supplement_1, July 2021, Pages i25–i33, https://doi.org/10.1093/bioinformatics/btab293

Jing-Zhe Jiang, Wen-Guang Yuan, Jiayu Shang, Ying-Hui Shi, Li-Ling Yang, Min Liu, Peng Zhu, Tao Jin, Yanni Sun, Li-Hong Yuan, Virus classification for viral genomic fragments using PhaGCN2, Briefings in Bioinformatics, 2022;, bbac505, https://doi.org/10.1093/bib/bbac505
```

