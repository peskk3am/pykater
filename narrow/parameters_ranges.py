import sys
import os

from db_connect import *

from methods import *

import numpy as np 


datasets = [
    12,
    14,
    16,
    18,  
    20,    
    54,
    60,
    36,
    46,
    22,
    23,
    181,
    183,
    311,
    313,
    307,
    1038,
    40670,
    1041,
    
    53,
]    

#    62,
#    39,
#    40,
#    41,
#    43,
#    1104,
#    212,
#    61
#]


methods = [
    "AdaBoostClassifier",
    "BernoulliNB",      
    "DecisionTreeClassifier",
    "GradientBoostingClassifier",
    "KNeighborsClassifier",
    "LinearDiscriminantAnalysis",
    "LinearSVC", 
    "PassiveAggressiveClassifier",
    "QuadraticDiscriminantAnalysis",   
    "RandomForestClassifier", 
    "SGDClassifier"     
    ]
    

methods_dict = {
    "AdaBoostClassifier": "adaboost",
    "BernoulliNB": "bernoulli_nb",      
    "DecisionTreeClassifier": "decision_tree",
    "GradientBoostingClassifier": "gradient_boosting",
    "KNeighborsClassifier": "knn",
    "LinearDiscriminantAnalysis": "lda",
    "LinearSVC": "linear_svc", 
    "PassiveAggressiveClassifier": "passive_aggressive",
    "QuadraticDiscriminantAnalysis": "qda",   
    "RandomForestClassifier": "random_forest", 
    "SGDClassifier": "sgd"     
    }


preprocessings = ["[]",
                  "['pca']",
                  "['normalize']",
                  "['scale']",
                  "['map_to_gaussian']",           
                  "['map_to_uniform']"
                  ]


preprocessings_labels = {
    "[]": "no preproc.",
    "['pca']": "PCA",
    "['normalize']": "Normalizer",
    "['scale']": "Standard Scaler",
    "['map_to_gaussian']": "Power Transformer",           
    "['map_to_uniform']": "Quantile Transformer"
    }

    
def get_data_from_db(dataset, method, pre, threshold=1):
    
    if mydb:
        mycursor = mydb.cursor()                                          
        
        val = (dataset, methods_dict[method], pre, 1-threshold)
        
        sql = "SELECT accuracy, parameters FROM results WHERE dataset=%s AND method=%s AND preprocessings=%s AND accuracy>%s"
        mycursor.execute(sql, val)
        
        
        myresult = mycursor.fetchall()
        
        errors = []  # [(error, params_dict)]
        try:
            for x in myresult:
                errors.append( (1-float(x[0]), eval(x[1])) )
        except:
            pass
                
        return errors    


def get_threshold_from_db(dataset, method, pre="[]"):
    
    if mydb:
        mycursor = mydb.cursor()                                          
        
        val = (dataset, methods_dict[method], pre)
        
        sql = "SELECT accuracy FROM results WHERE dataset=%s AND method=%s AND preprocessings=%s AND accuracy IS NOT NULL AND search LIKE 'rand%'"
        mycursor.execute(sql, val)
        
        myresult = mycursor.fetchall()
        
        errors = []  # [(error, params_dict)]
        try:
            for x in myresult:
                errors.append( (1-float(x[0])) )
        except:
            pass
                
        return errors    



threshold_percentile = 25

params_dict = {}  # { method: hyperparameter space}

all_params_dict = {}  # { param_name: [lower, upper, lower_tuned, upper_tuned] or [choices, choices_tuned] }
dataset_params_dict = {}  # { dataset: { param_name: ... } }

all_params_dict_min_only = {}  # { param_name: [lower, upper] or [choices] }


def create_instances(d, m):
    X = []
    y = []
    
    # set threshold - x th percentil    
    thresh_errors = get_threshold_from_db(d, m, "[]")
                             
    if len(thresh_errors) == 0:
        return
        
    threshold = np.percentile(thresh_errors, threshold_percentile)
    minimum =  min(thresh_errors)        
        
    print(m, d, threshold, minimum)
    

    errors_params = get_data_from_db(d, m, "[]", threshold)  # [(error, params_dict)]
    
    f = open(m+".txt", "a")
    
    error, params = errors_params[0]
    
    # sort keys
    sorted_keys = sorted(params.keys())
    
    if d == 12:
        # write header
        f.write(str(sorted_keys)[1:-1]+", error\n")
    
    
    file_name = m+"_"+str(d)+".txt"
    fd = open(file_name, "w")
    fd.write(str(sorted_keys)[1:-1]+", error\n")
    
    for error, params in errors_params:
        
        x = [ params[key] for key in sorted_keys ] 
        y = error
        
        
        x_str = str(x)[1:-1]         
        
        f.write(x_str+", "+str(y)+"\n")
        fd.write(x_str+", "+str(y)+"\n")
    fd.close()
    
    f.close()
    
    
            

def create_dataset(m):
    f = open(m+".txt", "w")
    f.close()    
    for d in datasets:
        create_instances(d, m)
                   
                   

def get_params_values(d, m):
    global params_dict, all_params_dict, dataset_params_dict

    # set threshold - x th percentil    
    thresh_errors = get_threshold_from_db(d, m, "[]")

    errors_params = get_data_from_db(d, m, "[]")  # [(error, params_dict)]
    
                         
    if len(thresh_errors) == 0:
        return
        
    threshold = np.percentile(thresh_errors, threshold_percentile)
    minimum =  min(thresh_errors)        
        
    print(m, d, threshold, minimum)
    
    
    
    temp_min_dict = {}
    min_error = 1
    best_count = 0        
    for error, params in errors_params:
        # get param type, range
        hs = params_dict[m]        
            
        if error <= threshold:
        
            change_min_dict = False
            if error < min_error:
                min_error = error
                change_min_dict = True
                     
            # print(error, params)
            # print("------------------------")
            for param_name in params:
                hp = hs.get_hyperparameter_by_name(param_name)                
                value = params[param_name]
                # add value to the hyper parameter
                if type(hp).__name__ in ["FloatHyperparameter", "IntegerHyperparameter"]:                 
                    # save values for each method, dataset pair
                    if value < hp.lower_tuned:  
                        hp.lower_tuned = value
                        
                    if value > hp.upper_tuned:  
                        hp.upper_tuned = value

                    # save values for each parameter
                    if value < all_params_dict[hp.name][2]:  
                        all_params_dict[hp.name][2] = value
                        
                    if value > all_params_dict[hp.name][3]:  
                        all_params_dict[hp.name][3] = value

                    # save values for each parameter for each dataset                    
                    if value < dataset_params_dict[d][hp.name][2]:                      
                        dataset_params_dict[d][hp.name][2] = value
                        
                    if value > dataset_params_dict[d][hp.name][3]:  
                        dataset_params_dict[d][hp.name][3] = value

                    if change_min_dict:
                        temp_min_dict[hp.name] = [value, value]
                        

                if type(hp).__name__ in ["CategoricalHyperparameter"]:
                    
                    if value not in hp.choices_tuned:
                        hp.choices_tuned += [value]
                                            
                    if value not in all_params_dict[hp.name][1]:    
                        all_params_dict[hp.name][1] += [value]
                    
                    if value not in dataset_params_dict[d][hp.name][1]:    
                        dataset_params_dict[d][hp.name][1] += [value]
                                                                        
                    if change_min_dict:
                        temp_min_dict[hp.name] = [value]
    
    
    # "add" temp dict to all_params_dict_min_only
    for param_name in temp_min_dict:
        item = temp_min_dict[param_name]
        value = item[0]
        if len(item) == 2:                
            if value < all_params_dict_min_only[param_name][0]:                      
                all_params_dict_min_only[param_name][0] = value
                        
            if value > all_params_dict_min_only[param_name][1]:  
                all_params_dict_min_only[param_name][1] = value
                
        if len(item) == 1:
            if value not in all_params_dict_min_only[param_name]:
                all_params_dict_min_only[param_name] += [value]
                

def compute_intersection():     
     # dataset_params_dict = {}  # { dataset: { param_name: ... } }

     params_dict_inter = {}
                    
     for d in datasets:         
         for param_name in dataset_params_dict[d]:
             if len(dataset_params_dict[d][param_name]) == 4: 
                 if param_name not in params_dict_inter:
                     # put in the first interval
                     params_dict_inter[param_name] = dataset_params_dict[d][param_name]
                 else:                
                     # find intersection
                     lower_d = dataset_params_dict[d][param_name][2]
                     upper_d = dataset_params_dict[d][param_name][3]
    
                     lower = params_dict_inter[param_name][2]
                     upper = params_dict_inter[param_name][3]
                      
                     # take the larger lower and the smaller upper
                     lower = max(lower_d, lower) 
                     upper = min(upper_d, upper)
                     
                     if lower < upper:  # intersect
                        params_dict_inter[param_name][2] = lower
                        params_dict_inter[param_name][3] = upper
                     else: # union
                        lower = min(lower_d, lower) 
                        upper = max(upper_d, upper)
                        params_dict_inter[param_name][2] = lower
                        params_dict_inter[param_name][3] = upper
                            
             else:
                 pass # TODO categorical
     
     print("INTERSECTION:")
     for param in params_dict_inter:
         print(param, params_dict_inter[param])
     print()            

def narrow_intervals():
    global params_dict, all_params_dict, dataset_params_dict
    
    for d in datasets:
        dataset_params_dict[d] = {}
    
                           
    for m in methods:                
        # init params dict
        method_file_name = methods_dict[m]
        hs = eval(method_file_name+".get_hyperparameter_search_space()")            
        
        for hp in hs.hyperparameters:
            # print(type(hp).__name__, hp.name)
            if type(hp).__name__ in ["FloatHyperparameter", "IntegerHyperparameter"]:
                hp.lower_tuned = float('inf')
                hp.upper_tuned = float('-inf') 
                hp_value = [hp.lower, hp.upper, hp.lower_tuned, hp.upper_tuned]
                
                for d in datasets:            
                    dataset_params_dict[d][hp.name] = [hp.lower, hp.upper, float('inf'), float('-inf')]
                
                all_params_dict_min_only[hp.name] = [float('inf'), float('-inf')] 
                
            if type(hp).__name__ in ["CategoricalHyperparameter"]:
                hp.choices_tuned = []
                hp_value = [hp.value, hp.choices_tuned] 
               
                for d in datasets:            
                    dataset_params_dict[d][hp.name] = [hp.value, []]
                
                all_params_dict_min_only[hp.name] = []    
                
            all_params_dict[hp.name] = hp_value
            
             
                                                    
            params_dict[m] = hs        

    
    
    for m in methods:
        for d in datasets:                                  
            get_params_values(d,m)
    
    for d in datasets:
        print(d, dataset_params_dict[d])
    
    
    # intersection of intervals
    compute_intersection()
                           

def init_dict(dictionary):
    # init params dict
    # { param_name: [lower_tuned, upper_tuned] or choices_tuned }
    for key in all_params_dict:
        param = all_params_dict[key]
        if len(param) == 4:
            dictionary[key] = [float('inf'), float('-inf')]
        if len(param) == 2:  
            dictionary[key] = [[]]
                    
    return dictionary
        

def verify():
    # verify using leave one out method    
    global params_dict, all_params_dict, dataset_params_dict
    
    for d_out in datasets:
        print()
        print("*************************")
        print("COMPARING DATASET", d_out)
        
        # count intervals for all except d
        
        all_params_temp = {}  # { param_name: [lower_tuned, upper_tuned] or choices_tuned }
        d_out_params_temp = {}
        
        all_params_temp = init_dict(all_params_temp)
        d_out_params_temp = init_dict(d_out_params_temp)
        
        # init 
        
        for d in datasets:
            if d != d_out:
                dictionary = all_params_temp
            else:
                dictionary = d_out_params_temp
                
            for p in dataset_params_dict[d]:
                param = dataset_params_dict[d][p]
                 
                if len(param) == 4: # numeric
                    lower_tuned = param[2]
                    upper_tuned = param[3]
                    
                    if lower_tuned < dictionary[p][0]:
                        dictionary[p][0] = lower_tuned
                        
                    if upper_tuned > dictionary[p][1]:
                        dictionary[p][1] = upper_tuned
   
                if len(param) == 2: # categorical                
                    choices_tuned = param[1]
                    for cht in choices_tuned:
                        if cht not in dictionary[p][0]:
                            dictionary[p][0] += [cht] 

                                                               
        # compare
        for p in all_params_temp:
                
            param = all_params_temp[p]
            if len(param) == 2: # numeric
                lower_tuned = param[0]
                upper_tuned = param[1]
                                
                if lower_tuned <= d_out_params_temp[p][0]:
                    # ok, test passed
                    print("Dataset", d_out, p, "lower: OK", end=" ")                     
                                                  
                else:
                    # test failed, by how much
                    difference = d_out_params_temp[p][0]-lower_tuned                    
                    print("Dataset", d_out, p, "lower:", lower_tuned, "d_out:", d_out_params_temp[p][0], "diff:", difference, end=" ")
                    
                if upper_tuned >= d_out_params_temp[p][1]:
                    # ok, test passed
                    # print("Dataset", d_out, p, "upper: OK")
                    print("upper: OK")
                                                            
                else:                
                    # test failed, by how much
                    difference = d_out_params_temp[p][1]-upper_tuned
                    print("upper: ", upper_tuned, "d_out:", d_out_params_temp[p][1], "diff:", difference)
                    
            if len(param) == 1: # categorical                
                choices_tuned = param[0]
                
                difference = []                                
                for cht in d_out_params_temp[p][0]: 
                    if cht not in choices_tuned:
                         difference += [cht]
                        
                if len(difference) == 0:
                    # ok, test prosel
                    print("Dataset", d_out, p, "choices: OK")                        
                                        
                else:
                    # test failed, by how much        
                    print("Dataset", d_out, p, "choices", choices_tuned, "d_out:", d_out_params_temp[p][0], "diff:", difference)                                         


narrow_intervals()
# verify()       
 

for param_name in all_params_dict_min_only:
    print(param_name, all_params_dict_min_only[param_name])


#  all_params_dict = {}  # { param_name: [lower, upper, lower_tuned, upper_tuned] or [choices, choices_tuned] }
print()
print("*********************")
print()

f_numeric = open("params_numeric_"+str(threshold_percentile), "w")
f_categorical = open("params_categorical_"+str(threshold_percentile), "w")

f_numeric.write("method param lower upper lower_tuned upper_tuned\n")
f_categorical.write("method param choices choices_tuned\n")

for param_name in all_params_dict:
    print(param_name, all_params_dict[param_name])

    params = all_params_dict[param_name]
    
    method, p = param_name.split("__")   
    if len(params) == 2:
        choices = params[0]
        choices_tuned = params[1]
        
        line = method+" "+p+" "+str(choices)+" "+str(choices_tuned)
        f_categorical.write(line+"\n")
         
    if len(params) == 4:
        lower = params[0]
        upper = params[1]
        lower_tuned = params[2]
        upper_tuned = params[3]
        
        line = method+" "+p+" "+str(lower)+" "+str(upper)+" "+str(lower_tuned)+" "+str(upper_tuned)
        f_numeric.write(line+"\n") 
             

f_numeric.close()
f_categorical.close()

