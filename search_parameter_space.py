import sklearn.model_selection as model_selection
from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

from methods import *

import ea_search
import load_openml_datasets

#----------
#  List of used data-mining methods
#----------
# list of names of the modules containing methods (in 'methods' directory)
methods = ["knn", "decision_tree"]

# Load the iris dataset from scikit-learn
# from sklearn import datasets
# iris = datasets.load_iris()
# X, y = iris.data, iris.target


#----------
#  Load datasets from OpenML
#----------
# use first 3 datasets
datasets = load_openml_datasets.get_datasets(first_n=3) # list of tuples (X,y)


#----------
#  Search algorithms
#----------

def grid_search(m):
    tuned_parameters = m.grid_hyperparameters
    print(tuned_parameters)
    model = model_selection.GridSearchCV(m.get_model_class()(), tuned_parameters)    
    return model 

def evolutionary_search(m):
    model = ea_search.EvolutionSearchCV(m.get_model_class()(),
        m.get_hyperparameter_search_space())    
    return model

def evolutionary_search_missing_values(m):
    model = Pipeline([("imputer", Imputer(missing_values=0,
                                          strategy="mean",
                                          axis=0)),
                      ('ea', ea_search.EvolutionSearchCV(m.get_model_class()(),
                                m.get_hyperparameter_search_space()))])  
    return model
    
    
#----------
#  Main cycle: for each dataset go through all methods
#----------
for X,y in datasets:

    print("X: ", X)
    print("y: ", y)
  
    # Split the dataset into testing and training data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=0)
        
    for m in methods:
        m = eval(m)
        print("--- ", m.get_name())    
        
        #----------
        #  Select the search algorithm
        #----------

        # Use scikit-learn grid search
        # model = grid_search(m)    TODO ted nefunguje
        
        # Use deap evolutionary search 
        # model = evolutionary_search(m)
        
        # Use deap evolutionary search with replaced missing values
        model = evolutionary_search_missing_values(m)
        
        #----------
        #  Train the search
        #----------        
        model.fit(X_train, y_train)
        
        #----------
        #  Evaluate the search
        #----------
        print()
        print("Detailed classification report:")        
        y_true, y_pred = y_test, model.predict(X_test)
      
        print(classification_report(y_true, y_pred))
        print()
        print()
