import sys
import os

from db_connect import *

from numpy import nan, isnan, mean, median, percentile 

import matplotlib.pyplot as plt
import numpy as np 


# ----- begin config ------

'''
    CONFIG:
    
        - print data to file "pre_random_benchmark_"+method+"_data.txt"
            print_data_table         True/False            
'''

print_data_table = True

# ----- end config ------


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
    62,
    39,
    40,
    41,
    43,
    1104,
    212,
    61
]


method_list = [ "BernoulliNB",
                "KNeighborsClassifier",
                "DecisionTreeClassifier",
                "SGDClassifier",
                "PassiveAggressiveClassifier",
                "LinearDiscriminantAnalysis",
                "QuadraticDiscriminantAnalysis",
                "LinearSVC",
                "AdaBoostClassifier",
                "GradientBoostingClassifier",
                "RandomForestClassifier"
]

methods_short = {"BernoulliNB": "bernoulli_nb",
                "KNeighborsClassifier": "knn",
                "DecisionTreeClassifier": "decision_tree",
                "SGDClassifier": "sgd",
                "PassiveAggressiveClassifier": "passive_aggressive",
                "LinearDiscriminantAnalysis": "lda",
                "QuadraticDiscriminantAnalysis": "qda",
                "LinearSVC": "linear_svc",
                "AdaBoostClassifier": "adaboost",
                "GradientBoostingClassifier": "gradient_boosting",
                "RandomForestClassifier": "random_forest"
		}


labels_indexes = []
datasets_info = {}

def my_sort(data):
    
    global labels_indexes
    
    criteria = []  # [(method_index, median)]
    criteria1 = []  # [(method_index, max-min)]
    i = 0  # method index  
    for d in data:
       i += 1
       if len(d) == 0:
           continue
           
       criteria.append((i-1, -np.median(d)))
       criteria1.append( (method_list[i-1], round(min(d),2), round(percentile(d,25),2), round(median(d),2), round(percentile(d,75),2), round(max(d),2), round(max(d) - min(d),2)) )       

       
    criteria.sort(key=lambda x: x[1])
    criteria1.sort(key=lambda x: x[-1])
  
    data_new = []
    
    if print_data_table:
        f = open("all_methods_tex_table.txt", 'w')
        # write header
        f.write("Method & Min & 25\\textsuperscript{th} Perc. & Median & 75\\textsuperscript{th} Perc. & Max & Err. Rng\\\\\n")
        f.write("\\hline\\hline\n")
        for c in criteria1:                        
            
            values_str = str(c).strip("()").replace(", ", "&").replace("_","\\_")
            f.write(values_str+"\\\\\n")
            # draw a line
            f.write("\\hline\n")
        f.close()
    
    
    for c in criteria1:
        print(c)
    
    print()
  
    for c in criteria:
        print(c)
        data_new.append(data[c[0]])
        labels_indexes.append(c[0])
    
    return data_new      


def plot(data):        
    # data ... sequence of vectors
    
    data = my_sort(data)
    
    # create labels
    labels = []
    for i in labels_indexes:
        labels.append(method_list[i])         
        
        
    fig1, ax1 = plt.subplots(figsize=(10, 10))    
    ax1.set_title("Error ranges for all datasets")
    
    
    bp = ax1.boxplot(data, vert=False)
        
    ax1.set_ylabel('')
    print(len(labels), len(data))
    
    plt.yticks(range(1, len(labels)+1), labels)
    plt.subplots_adjust(left=0.35)
    
    plt.savefig('all_methods.png')
    plt.savefig('all_methods.pdf')
    plt.close(fig1)    
        

def get_data_from_db(method, dataset):
    
    if mydb:
        mycursor = mydb.cursor()                                          
        
        val = (methods_short[method], dataset)
        
        sql = "SELECT accuracy FROM results WHERE method=%s AND dataset=%s AND accuracy IS NOT NULL AND preprocessings='[]'"
        mycursor.execute(sql, val)
        # print(mycursor.statement)

        myresult = mycursor.fetchall()
        
        errors = []
        try:
            for x in myresult:
                errors.append( 1-float(x[0]) )
        except:
            pass
            
        value = None                
        
        if len(errors) > 0: 
            value = np.percentile(errors, 75) - np.percentile(errors, 25) 
                
        return value    


def prepare_data():
    data = []
    for m in method_list:
        data_m = []
        for d in datasets:
            value = get_data_from_db(m, d)
            if value:
                data_m.append(value)
        data.append(data_m)

    return data    


data = prepare_data()    
plot(data)
