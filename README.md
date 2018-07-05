# Collaborative Filtering 
Semestral project of Computational Intelligence Lab at ETH Zürich

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
