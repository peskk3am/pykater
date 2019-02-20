from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
import sklearn.model_selection as model_selection

from methods import *

import ea_search
import load_openml_datasets

import sys


#----------
#  Get command line args
#----------

method_index = int(sys.argv[1])    # 0 - 14
dataset_index = int(sys.argv[2])   # 0 - 9



#----------
#  List of used data-mining methods
#----------
# list of names of the modules containing methods (in 'methods' directory)
methods = ["knn", "decision_tree", "gaussian_nb", "adaboost",
           "linear_svc", "sgd", "multinomial_nb", "passive_aggressive",
           "random_forest", "bernoulli_nb", "svm_svc", "extra_trees",
           "gradient_boosting", "lda", "qda"]

methods = [methods[method_index]]

# Load the iris dataset from scikit-learn
# from sklearn import datasets
# iris = datasets.load_iris()
# X, y = iris.data, iris.target


#----------
#  Load datasets from OpenML
#----------
# # datasets = load_openml_datasets.get_datasets(first_n=1) # list of tuples (X,y)

# datasets = load_openml_datasets.get_10_liked_datasets(dataset_index)
datasets = load_openml_datasets.get_dataset(dataset_index)


#----------
#  Search algorithms
#----------

def grid_search(m):
    tuned_parameters = m.get_hyperparameter_search_space().grid_hyperparameters
    print(tuned_parameters)
    model = model_selection.GridSearchCV(m.get_model_class()(), tuned_parameters, cv=3)    
    return model 

def evolutionary_search(m):
    model = ea_search.EvolutionSearchCV(m.get_model_class()(),
        m.get_hyperparameter_search_space())    
    return model

def evolutionary_search_missing_values(m, dataset_name):
    model = Pipeline([("imputer", Imputer(missing_values=0,
                                          strategy="mean",
                                          axis=0)),
                      ('ea', ea_search.EvolutionSearchCV(
                                m.get_model_class()(),
                                m.get_hyperparameter_search_space(),
                                dataset_name))])  
    return model
    
    
#----------
#  Main cycle: for each dataset go through all methods
#----------
for X,y, dataset_name in datasets:

    #print("X: ", X)
    #print("y: ", y)
    # Split the dataset in two equal parts
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
        model = grid_search(m)
        
        #----------
        #  Train the search
        #---------- 
        try:       
            model.fit(X_train, y_train)
            model.score(X_test, y_test)
        except Exception as e:      
            print("Exception: "+str(e))
                        
   
        print("Best parameters set found on development set:")
        print()
        print(model.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = model.cv_results_['mean_test_score']
        stds = model.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, model.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
    
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()


        # evaluate:
        from sklearn.ensemble import RandomForestClassifier
        estimator = RandomForestClassifier()
        for params in model.cv_results_['params']:
            score = 0
            n = 10
            for k in range(n):
                estimator.__init__() 
                estimator.set_params(**params)
                # print(self.estimator.get_params())            
                estimator.fit(X_train, y_train)
                s = estimator.score(X_test, y_test)
                #print(s)
                score += s
            score = score / n 
            print(score)    