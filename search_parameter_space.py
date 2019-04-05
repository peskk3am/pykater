from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

from preprocessings import *
from methods import *

import ea_search, grid_search, randomized_search, simulated_annealing_search
import load_openml_datasets
import openml_datasets

import sys

from _config import *

if verbose > 0:
    print("Verbose:", verbose, "\n")

#----------
#  List of used space search methods
#----------
# list of names of the modules containing methods (in 'methods' directory)
search_list = ["grid_search_cv",
               "evolutionary_search", "evolutionary_search_cv",
               "randomized_search_cv", "simulated_annealing_search_cv"]

# grid_search_cv ... Use scikit-learn grid search                
# evolutionary_search_no_missing_values ... Use deap evolutionary search         
# evolutionary_search ... Use deap evolutionary search with replaced 
#                         missing values
# evolutionary_search_cv ... Use deap evolutionary search with replaced 
#                            missing values and with cross validation        
# randomized_search_cv ... Use scikit-learn randomized search 
        

#----------
#  List of available data-mining methods
#----------
# list of names of the modules containing methods (in 'methods' directory)
method_list = ["knn", "decision_tree", "gaussian_nb", "adaboost",
           "linear_svc", "sgd", "passive_aggressive",
           "random_forest", "bernoulli_nb", "svm_svc", "extra_trees",
           "gradient_boosting", "lda", "qda"]


#----------
#  List of preprocessings
#----------
# list of names of the modules containing methods (in 'methods' directory)
preproc_list = ["pca", "scale", "normalize", "map_to_uniform", "map_to_gaussian"]


#----------
#  Get command line args
#----------
 
if sys.argv[1] == "?":
    print()
    print("Usage: search_parameter_space.py search preprocessings-method-chain dataset")
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

    print("Preprocessings:")
    i = 0
    for p in preproc_list: 
        print("\t", i, "-", preprocessing_list[i])
        i += 1 
    print()
    print()

    print("Dataset: OpenML dataset ID")
    print("\t(iris id is 61)")     
    
    sys.exit()

search_index = int(sys.argv[1])
preprocs_method_chain = sys.argv[2]    # 0 - 9
dataset_index = int(sys.argv[3])

if "-" in preprocs_method_chain:
    preprocs_method = preprocs_method_chain.split("-")    
else:  
    preprocs_method = [preprocs_method_chain]
    
method_index = preprocs_method[-1]    
preprocs_indices = preprocs_method[:-1]  # all but the last

preprocs = [preproc_list[int(p)] for p in preprocs_indices] 
method = method_list[int(method_index)]  # TODO - only one dataset for now

search = search_list[search_index]

if verbose > 0:
    print ("Search:", search)
    print ("Methods:", method)    
    print ("Preprocs:", preprocs)
    
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


# TEMP solution:
did = openml_datasets.classification[dataset_index-1]
datasets = load_openml_datasets.get_dataset(did) 

dataset_index = did

#----------
#  Search algorithms
#----------

def grid_search_cv(pipeline, chain_names, chain_hyperparameter_space, dataset_name, verbose=0):
            
    tuned_parameters = chain_hyperparameter_space.get_grid_parameters()
    
    if verbose > 0:
        print("Parameters : values to test")
        for k in tuned_parameters:
          print("  --", k, ":", tuned_parameters[k])                              
        
    """ 
        Grid search from scikit-learn:
    
        GridSearchCV(estimator, param_grid, scoring=None, fit_params=None, 
        n_jobs=1, iid=True, refit=True, cv=None, verbose=0, 
        pre_dispatch=‘2*n_jobs’, error_score=’raise’, return_train_score=’warn’)
        
        param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of parameter
        settings to try as values, or a list of such dictionaries, in which case
        the grids spanned by each dictionary in the list are explored. This 
        enables searching over any sequence of parameter settings.        
    """    
    
    model = grid_search.GridSearch(pipeline, chain_names, tuned_parameters,
                                   dataset_name, verbose=0,)
                                          
    return model 

def evolutionary_search(m, dataset_name, verbose=0):     
    print ("evolutionary_search no longer supported. Use evolutionary_search_cv instead.")
    return

    model = Pipeline([("imputer", Imputer(missing_values=0,
                                          strategy="mean",
                                          axis=0)),
                      ('ea', ea_search.EvolutionarySearch(
                                m.get_model_class()(),
                                m.get_hyperparameter_search_space(),
                                dataset_name,
                                verbose=verbose))])                                     
    return model
                                
def evolutionary_search_cv(pipeline, chain_names, chain_hyperparameter_space, dataset_name, verbose=0):
       
    model = Pipeline([("imputer", Imputer(missing_values=0,     # TODO jo????
                                          strategy="mean",
                                          axis=0)),
                      ('ea', ea_search.EvolutionarySearchCV(
                                pipeline,
                                chain_names,
                                chain_hyperparameter_space,
                                dataset_name,                                
                                verbose=verbose))])
                                                           
    return model
    
def randomized_search_cv(pipeline, chain_names, chain_hyperparameter_space, dataset_name, verbose=0):
        
    """ Randomized search from scikit-learn    
    """    
    
    model = randomized_search.RandomizedSearch(pipeline, chain_names,
                                   chain_hyperparameter_space,
                                   dataset_name, verbose=verbose)                                          
    return model 


def simulated_annealing_search_cv(pipeline, chain_names, chain_hyperparameter_space, dataset_name, verbose=0):
        
    """ Simulated annealing search    
    """    
    
    model = simulated_annealing_search.SimulatedAnnealingSearch(pipeline,
                                   chain_names,
                                   chain_hyperparameter_space,
                                   dataset_name, verbose=verbose)                                          
    return model 


    
#----------
#  Main cycle: for each dataset go through all methods
#----------
for X,y, dataset_name in datasets:

    #print("X: ", X)
    #print("y: ", y)

    #----------
    #  Get preprocessing(s) method chain
    #----------
                  
    print("---")    
    p_names = [eval(p).get_name() for p in preprocs]  # jenom kvuli vypisu   
    print("Preprocessings:")    
    [print("---", p) for p in p_names]
    
    m = eval(method)
                
    print("Method:", m.get_name(), "(", method ,")")    
    print("---")
    
    # create the chain
    # example: Pipeline(steps=[('pca', pca), ('logistic', logistic)])
    p_chain = [(p, eval(p).get_model_class()() ) for p in preprocs]
    m_chain = [(method, m.get_model_class()() )]
    chain = p_chain + m_chain
    
    # print (chain)    
    
    pipeline = Pipeline(steps=chain)
    
    chain_names = preprocs+[method]
    
    # join parameters spaces of all chain items
    chain_hyperparameter_space = hyperparameters.HyperparameterSpace()
    for name in chain_names:
        # join all hyperparams spaces         
        hs = eval(name).get_hyperparameter_search_space()
        chain_hyperparameter_space.add_hyperparameter_space(hs)                       
 
    
    #----------
    #  Select the search algorithm
    #----------

    # construct function name from args
    # grid_search_cv(...), evolutionary_search(...) ...
    
    create_model_function = eval(search)         
    model = create_model_function(pipeline, chain_names, chain_hyperparameter_space, str(dataset_name), verbose=verbose)
                     
    
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
        
