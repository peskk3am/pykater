1) Quick-install MINICONDA
  * http://conda.pydata.org/docs/install/quick.html

2) Create an environment
  * specify packages you want to install:
        
```conda create --name python-pikater python=3 numpy scipy scikit-learn pandas deap```

3) Activate the environment
        
```activate python-pikater``` 

4) Install deap   
        
```pip install deap```   

5) Install OpenML python package 
   * github openml -> master branch, install:   
        
```git clone https://github.com/openml/openml-python```
   
```cd openml-python```

```python setup.py install```
   
(https://github.com/openml/openml-python/blob/develop/examples/OpenMLDemo.ipynb)

6) ```python search_parameter_space.py```
  
----
Conda guide:
  * http://conda.pydata.org/docs/test-drive.html#managing-conda

```conda info --envs```
