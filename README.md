# LDL-LRR
Re-implementation of "Label Distribution Learning by Maintaining Label Ranking Relation"

LDL-LRR: a **L**abel **D**istribution **L**earning algorithm by maintaining **L**abel **R**anking **R**elation 
 
## Environment
python=3.7.6, numpy=1.21.6, scipy=1.7.3, pytorch=1.13.0+cpu.

## Reproducing
Change the directory to this project and run the following command in terminal.
```Terminal
python demo.py
```


## Usage
Here is a simple example of using LDL-LRR.
```python
from utils import report
from ldllrr import LDL_LRR
# load data
Xtrain, Xtest, Dtrain, Dtest = load_dataset('sj') # this api should be defined by users

# train the model
lrr = LDL_LRR().fit(Xtrain, Dtrain)
Dhat = lrr.predict(Xtest)

# show the performance
report(Dhat, Dtest)
```

## Datasets
- The datasets used in our work is partially provided by [PALM](http://palm.seu.edu.cn/xgeng/LDL/index.htm)
- Emotion6: [http://chenlab.ece.cornell.edu/people/kuanchuan/index.html](http://chenlab.ece.cornell.edu/people/kuanchuan/index.html)

## Paper
```latex
@article{Jia2023LDLLRR,
	author={Xiuyi Jia and Xiaoxia Shen and Weiwei Li and Yunan Lu and Jihua Zhu},
	journal={IEEE Transactions on Knowledge and Data Engineering}, 
	title={Label Distribution Learning by Maintaining Label Ranking Relation}, 
	year={2023},
	volume={35},
	number={2},
	pages={1695-1707}
}
```
