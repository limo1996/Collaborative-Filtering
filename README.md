# Collaborative Filtering 
Semestral project of Computational Intelligence Lab at ETH Zürich

## Competition and Data
This project was used for in-class kaggle [competition](https://www.kaggle.com/c/cil-collab-filtering-2018). Data about the users and items can be found in the [repo](https://github.com/limo1996/Collaborative-Filtering/blob/master/data/data_train.csv) or on [kaggle](https://www.kaggle.com/c/cil-collab-filtering-2018/data).

## Techniques used
We tried variety of techniques either implemented by us or with help of [Surprise](https://github.com/NicolasHug/Surprise) library. These methods are:
  * SVD
  * SVD++
  * NMF
  * SGD
  * ALS
  * Baseline
  * Blending
  
Best performances of baseline algorithms are summed up in following table:

| Model   | k   | η   | Public RMSE   |
| ------------- | ------------- | ----- | ---- |
| ALS | 900 | 0.5 | 1.07754 |
| SVD | 7 | -- | 1.05471 |
| NMF | 8 | 0.0055 | 0.99807 |
| Baseline* | --  | 0.005 | 0.99768 |
| SVD* | 1 | 0.005 | 0.99566 |
| SVD++ | 1 | 0.007 | 0.99507 |
| SGD | 12 | 0.00638 | 0.98288 |
| SGD<sub>lr</sub> | 12 | <0.09450,0.00005> | 0.97949 |

*Models marked with * were run with help of Surprise library.*

Our novel solution is based on SGD configurations blending. You can read more about it in our [paper](report/report.pdf). Results of best performing blending are reported in following table:

|Blended models   | Weights | Public RMSE   |
| ------------- | ------------- | ----- |
| SGD<sub>lr</sub>/SVD++/Baseline/SVD | 4/1/1/4 | 0.98842 |
| SGD<sub>lr</sub>/SVD++/Baseline | 4/1/1 | 0.98126 |
| SGD<sub>lr</sub>/SGD<sub>lr</sub>(12,0.075,0.04,2.9) | 5/4 | 0.97817 |
| SGD<sub>lr</sub>([10..17],0.08,0.04,2.9) | 1/1../1 | 0.97723 |
| SGD<sub>lr</sub>([10..17],0.08,0.04,2.9)/SGD<sub>lr</sub>(12,0.08,0.04,[2.5,2.6,..3.5]) | 1/1../1 | 0.97718 |

Where SGD<sub>lr</sub>(k,λ1,λ2,η) is the configuration of SGD<sub>lr</sub> and default  configuration is SGD<sub>lr</sub> =SGD<sub>lr</sub>(12,0.08,0.04,2.9). If one of the parameters is vector then we treat it as vector of configurations with only one parameter changing. 

## Reproducibility
The goal of this project was not to create user friendly library but rather come up with novel approach to collaborative filtering. Therefore we provide only wrapper script that can be used for reproducing the results. Exact steps for reproducing the code are as follows:

First we need to clone the repo and go to source folder:
```bash
git clone https://github.com/limo1996/Collaborative-Filtering.git
cd Collaborative-Filtering
cd src
```
When we are in source folder we can execute the wrapper script to reproduce the results:
  * Command `python3 main.py --bestSGD` reproduces submission file called **bestSGD.csv** with the best achieved RMSE for SGD.
  * Command `python3 main.py --gridSearch` runs grid search for the best SGD configuration.
  * Command `python3 main.py --bestBlending` creates submission file named **bestBlending.csv** with the best achieved RMSE for Blending. Note that this script does not run SGD for all configurations but loads already produced predictions from *data/var_k* folder. 
  * Command `python3 main.py --generatePlots` displays plots that you can see in the [report](report/report.pdf)
