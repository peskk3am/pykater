from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, normalize


from methods import *

import ea_search, grid_search, randomized_search
import load_openml_datasets

import sys

verbose = 3

if verbose > 0:
    print("Verbose:", verbose, "\n")

#----------
#  List of used space search methods
#----------
# list of names of the modules containing methods (in 'methods' directory)
search_list = ["grid_search_cv", "evolutionary_search_no_missing_values",
               "evolutionary_search", "evolutionary_search_cv",
               "randomized_search_cv"]

# grid_search_cv ... Use scikit-learn grid search                
# evolutionary_search_no_missing_values ... Use deap evolutionary search         
# evolutionary_search ... Use deap evolutionary search with replaced 
#                         missing values
# evolutionary_search_cv ... Use deap evolutionary search with replaced 
#                            missing values and with cross validation        
# randomized_search_cv ... Use scikit-learn randomized search 
        

#----------
#  List of used data-mining methods
#----------
# list of names of the modules containing methods (in 'methods' directory)
method_list = ["knn", "decision_tree", "gaussian_nb", "adaboost",
           "linear_svc", "sgd", "multinomial_nb", "passive_aggressive",
           "random_forest", "bernoulli_nb", "svm_svc", "extra_trees",
           "gradient_boosting", "lda", "qda"]

#----------
#  Get command line args
#----------
 
if sys.argv[1] == "?":
    print()
    print("Usage: search_parameter_space.py search method dataset")
    print()
    print("Parameter-space search algorithms:")
    i = 0
    for s in search_list: 
        print("\t", i, "-", search_list[i])
        i += 1 
    print("\tUse config files to set the searchparameters.")
    print()

    print("Methods:")
    i = 0
    for m in method_list: 
        print("\t", i, "-", method_list[i])
        i += 1 
    print()
    print()

    print("Dataset: OpenML dataset ID")
    print("\t(iris id is 61)")     
    
    sys.exit()

search_index = int(sys.argv[1])
method_index = int(sys.argv[2])    # 0 - 9
dataset_index = int(sys.argv[3])


methods = [method_list[method_index]]  # TODO - only one dataset for now
search = search_list[search_index]

if verbose > 0:
    print ("Search:", search)
    print ("Methods:", methods)
    
# Load the iris dataset from scikit-learn
# from sklearn import datasets
# iris = datasets.load_iris()
# X, y = iris.data, iris.target


#----------
#  Load datasets from OpenML
#----------
# # datasets = load_openml_datasets.get_datasets(first_n=1) # list of tuples (X,y)

# datasets = load_openml_datasets.get_10_liked_datasets(dataset_index)
# TODO - only one dataset for now
datasets = load_openml_datasets.get_dataset(dataset_index) 

#----------
#  Search algorithms
#----------

def grid_search_cv(m, dataset_name, verbose=0):
    tuned_parameters = m.get_hyperparameter_search_space().grid_hyperparameters
    if verbose > 0:
        print("Parameters : values to test")
        for k in tuned_parameters:
          print("  --", k, ":", tuned_parameters[k])
        #print(tuned_parameters)                      
    
    """ Grid search from scikit-learn:
    
        GridSearchCV(estimator, param_grid, scoring=None, fit_params=None, 
        n_jobs=1, iid=True, refit=True, cv=None, verbose=0, 
        pre_dispatch=‘2*n_jobs’, error_score=’raise’, return_train_score=’warn’)
        
        param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of parameter
        settings to try as values, or a list of such dictionaries, in which case
        the grids spanned by each dictionary in the list are explored. This 
        enables searching over any sequence of parameter settings.        
    """    
    
    model = grid_search.GridSearch(m.get_model_class()(), tuned_parameters,
                                   dataset_name, verbose=0,)
                                          
    return model 

def evolutionary_search_no_missing_values(m, dataset_name, verbose=0):
    model = Pipeline([("imputer", Imputer(missing_values=0,
                                          strategy="mean",
                                          axis=0)),
                      ('ea', ea_search.EvolutionarySearch(
                                m.get_model_class()(),
                                m.get_hyperparameter_search_space(),
                                dataset_name,
                                verbose=verbose))])  
    return model

def evolutionary_search(m, dataset_name, verbose=0):     
    model = Pipeline([("imputer", Imputer(missing_values=1,
                                          strategy="mean",
                                          axis=0)),
                      ('ea', ea_search.EvolutionarySearch(
                                m.get_model_class()(),
                                m.get_hyperparameter_search_space(),
                                dataset_name,
                                verbose=verbose))])                                     
    return model
                                
def evolutionary_search_cv(m, dataset_name, verbose=0):
    model = Pipeline([("imputer", Imputer(missing_values=0,     # TODO jo????
                                          strategy="mean",
                                          axis=0)),
                      ('ea', ea_search.EvolutionarySearchCV(
                                m.get_model_class()(),
                                m.get_hyperparameter_search_space(),
                                dataset_name,                                
                                verbose=verbose))])                           
    return model
    
def randomized_search_cv(m, dataset_name, verbose=0):
        
    """ Randomized search from scikit-learn    
    """    
    
    model = randomized_search.RandomizedSearch(m.get_model_class()(),
                                   m.get_hyperparameter_search_space(),
                                   dataset_name, verbose=verbose)
                                          
    return model 

    
#----------
#  Main cycle: for each dataset go through all methods
#----------
for X,y, dataset_name in datasets:

    #print("X: ", X)
    #print("y: ", y)
        
    for m in methods:
        m = eval(m)
        print("---")
        print("Method:", m.get_name())    
        
        #----------
        #  Select the search algorithm
        #----------

        # construct function name from args
        # grid_search_cv(...), evolutionary_search(...) ...
        
        create_model_function = eval(search)         
        model = create_model_function(m, str(dataset_name), verbose=verbose)
                         
        
        #----------
        #  Train the search
        #----------                 
        model.fit(X, y)
        
                        
        ##----------
        ##  Evaluate the best estimator found by the search
        ##----------
        # print()
        # print("Detailed classification report:")        
        # y_true, y_pred = y_test, model.predict(X_test)
        #
        # print(classification_report(y_true, y_pred))
        print()
        print()
        
