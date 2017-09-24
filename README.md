# Graph Wavelets via Sparse Cuts

Implementation of graph wavelets via sparse cuts with some baselines, datasets and evaluation.

Evaluation is performed using IPython Notebook.

After code review some results may differ from those presented in the paper.

Scalability and approximation experiments:
-----------------------
https://nbviewer.jupyter.org/github/arleilps/sparse-wavelets/blob/master/synthetic-data.ipynb

Compression experiments:
-----------------------
https://nbviewer.jupyter.org/github/arleilps/sparse-wavelets/blob/master/compression-experiments.ipynb

Test:
------
To run the DOCTEST in lib/experiments.py set PYTHONHASHSEED=0 and then
```
python -m doctest lib/experiments.py -v
```
<br />

For more details, see the paper:  
[Graph Wavelets via Sparse Cuts ](http://arxiv.org/abs/1602.03320 "")  
Arlei Silva, Xuan-Hong Dang, Prithwish Basu, Ambuj K Singh, Ananthram Swami  
ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2016 (to appear). 

Arlei Silva (arlei@cs.ucsb.edu)

