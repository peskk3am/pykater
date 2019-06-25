import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

m = "AdaBoostClassifier"
d = 12

train = pd.read_csv(m+"_"+str(d)+"_clean.txt", delimiter = ',')

print(train.head())         

'''
>>> from sklearn.preprocessing import OneHotEncoder
>>> ohe = OneHotEncoder(sparse=False)
>>> hs_train_transformed = ohe.fit_transform(hs_train)
>>> hs_train_transformed
array([[0., 0., 0., ..., 1., 0., 0.],
       [0., 0., 1., ..., 0., 0., 0.],
       [0., 0., 0., ..., 1., 0., 0.],
       ...,
       [0., 0., 0., ..., 1., 0., 0.],
       [0., 0., 1., ..., 0., 0., 0.],
       [0., 0., 1., ..., 0., 0., 0.]])
As expected, it has encoded each unique value as its own binary column.
'''

print(train["\'adaboost__algorithm\'"].value_counts())


all_columns = train.columns.values

kinds = np.array([dt.kind for dt in train.dtypes])
is_num = kinds != 'O'
num_cols = all_columns[is_num]

cat_cols = all_columns[~is_num]


ohe = OneHotEncoder(sparse=False)

si = SimpleImputer(strategy='constant', fill_value=-1)



transformers = [('cat', ohe, cat_cols ),
                ('num', si, num_cols )]
ct = ColumnTransformer(transformers=transformers)
train_transformed = ct.fit_transform(train)



#X = train_transformed[:, [2,3]]
X = train_transformed
print("NEW SHAPE:", X.shape)
print(X)


# y_pred = KMeans(n_clusters=2).fit_predict(X)

plt.scatter(X[:, 2],X[:, 3], c=X[:, 4])
plt.savefig(m+".png")