import sys
import os

from db_connect import *

import matplotlib.pyplot as plt
from numpy import nan, isnan, mean, median, percentile 


# ----- begin config ------

'''
    CONFIG:
    
        - print statistics to file  (output is in file "all_stats.txt")
            print_all_stats     True/False    
            
'''

print_all_stats = True


# ----- end config ------



datasets = [
    12,
    16,
    18,
    311,
    313,
    43,
]

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


def write_stats(dataset, method, data):
    '''  (pres vsechny searche) - pro dataset, method, pre: 
      
         min, median, min_bez_pre, median_bez_pre, rozdil_min, rozdil_median 
    '''                           
    
    stats = []
                
    for d in data:   # serazeny podle preprocessings       
        if len(d) == 0:
            continue              
        stats += [(min(d), median(d))]       
       
    f = open("stats_error_ranges_pre.txt", 'a')
    
    if len(stats) > 0:
        min_no_pre = stats[0][0]
        median_no_pre = stats[0][1]
        for i, s in enumerate(stats[1:]):
                                    
            out = (dataset, method, preprocessings_labels[preprocessings[i+1]], 
                  s[0], s[1], min_no_pre, median_no_pre, min_no_pre-s[0], median_no_pre-s[1])              
             
            f.write(str(out)+"\n")    
        
    f.close()

    

def plot(dataset, method):        
    print("Plotting:", dataset, method)
    # data ... sequence of vectors
    data = []    
    for pre in preprocessings:         
        data.append(get_data_from_db(dataset, method, pre))

    if print_all_stats:
        write_stats(dataset, method, data)
        # return 
    
    # create labels
    labels = []
    for p in preprocessings:
        labels.append(preprocessings_labels[p])                      
        
    fig1, ax1 = plt.subplots(figsize=(6, 8))    
    ax1.set_title(method+", "+"OpenML id: "+str(dataset))
    ax1.boxplot(data)
    ax1.set_ylabel('Error')
    print(len(labels), len(data))
    
    plt.xticks(range(1, len(labels)+1), labels, rotation='vertical')
    plt.subplots_adjust(bottom=0.35)
    
    plt.savefig(method+"_"+str(dataset)+'_pre.png')
    plt.savefig(method+"_"+str(dataset)+'_pre.pdf')
    plt.close(fig1)    
        

def get_data_from_db(dataset, method, pre):
    
    if mydb:
        mycursor = mydb.cursor()                                          
        
        val = (dataset, methods_short[method], pre)
        
        sql = "SELECT accuracy FROM results WHERE dataset=%s AND method=%s AND accuracy IS NOT NULL AND preprocessings=%s"
        mycursor.execute(sql, val)
        
        myresult = mycursor.fetchall()
        
        errors = []
        try:
            for x in myresult:
                errors.append( 1-float(x[0]) )
        except:
            pass
                
        return errors    



if print_all_stats:
    f = open("stats_error_ranges_pre.txt", 'w')
    f.close()
    
for m in methods:
    for d in datasets:
        plot(d, m)
