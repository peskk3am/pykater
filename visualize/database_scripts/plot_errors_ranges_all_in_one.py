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

        - create statistics .tex table  ("all_stats_tex_table.txt")
            print_all_stats_tex_table     True/False
'''

print_all_stats = True
print_all_stats_tex_table = True


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

labels_indexes = []
datasets_info = {}

def my_sort(data):
    # sort by (max - min)
    global labels_indexes
    
    criteria = []  # [(datasets_index, max-min)]
    stats = {}  # [(openml_id, min, max, mean, 25percentil, 75percentil, max-min)]
    i = 0    # index in datasets and in data
    for d in data:
       i += 1
       if len(d) == 0:
           continue
       criteria.append((i-1, max(d) - min(d)))       
       stats[datasets[i-1]] = (round(min(d),2), round(percentile(d,25),2), round(median(d),2), round(percentile(d,75),2), round(max(d),2), round(max(d) - min(d),2))       
       
    criteria.sort(key=lambda x: x[1])
        
    data_sorted = []
    for c in criteria:
        print(c)
        data_sorted.append(data[c[0]])
        labels_indexes.append(c[0])

    stats_sorted = []
    for c in criteria:        
        stats_sorted.append(stats[datasets[c[0]]])        
            

    if print_all_stats:
        f = open("all_stats.txt", 'w')
        for s in stats_sorted:
            f.write(str(s)+"\n")
        f.close()

    if print_all_stats_tex_table:

        f = open("all_stats_tex_table.txt", 'w')
        # write header
        f.write("Task (Inst, Attrs, Classes) & Min & 25\\textsuperscript{th} Perc. & Median & 75\\textsuperscript{th} Perc. & Max & Err. Rng\\\\\n")
        f.write("\\hline\\hline\n")
        for index in labels_indexes:        
            name, instances, attributes, classes = datasets_info.get(datasets[index], ("???", "???", "???", "???"))
            f.write(name+" ("+str(instances)+", "+str(attributes)+", "+str(classes)+") &")
            
            s = stats[datasets[index]]
            values_str = str(s).strip("()").replace(", ", "&")
            f.write(values_str+"\\\\\n")
            # draw a line
            f.write("\\hline\n")
        f.close()

    
    f = open("labels_indexes.txt", 'w')
    f.write(str(labels_indexes))
    f.close()
    
    return data_sorted      

def plot():
        
    # data ... sequence of vectors
    data = []
    for d in datasets:
        data.append(get_data_from_db(d))
    
    data = my_sort(data)
    
    # create labels
    labels = []
    for i in labels_indexes:
        name, instances, attributes, classes = datasets_info.get(datasets[i], ("???", "???", "???", "???"))
        l = str(datasets[i])+" "+name+" ("+str(instances)+", "+str(attributes)+", "+str(classes)+")" 
        labels.append(l)         
        
        
    fig1, ax1 = plt.subplots(figsize=(12, 9))    
    ax1.set_title("Errors ranges - all methods")
    ax1.boxplot(data)
    ax1.set_ylabel('Error')
    print(len(labels), len(data))
    
    plt.xticks(range(1, len(labels)+1), labels, rotation='vertical')
    plt.subplots_adjust(bottom=0.35)
    
    plt.savefig('all.png')
    plt.savefig('all.pdf')
    plt.close(fig1)    
        

def get_data_from_db(dataset):
    
    if mydb:
        mycursor = mydb.cursor()                                          
        
        val = (dataset,)
        
        sql = "SELECT accuracy FROM results WHERE dataset=%s AND accuracy IS NOT NULL AND preprocessings='[]'"
        mycursor.execute(sql, val)
        
        myresult = mycursor.fetchall()
        
        errors = []
        try:
            for x in myresult:
                errors.append( 1-float(x[0]) )
        except:
            pass
                
        return errors    


def get_dataset_names_from_db():
    
    if mydb:
        mycursor = mydb.cursor()                                          
                
        
        sql = "SELECT openml_id, name, instances, attributes, classes FROM datasets"
        mycursor.execute(sql)
        
        myresult = mycursor.fetchall()
        
        datasets_info = {}  # [openml_id: (name, instances, attributes, classes)]
        try:
            for x in myresult:
                datasets_info[x[0]] = x[1:]
        except:
            pass
                
        return datasets_info    

datasets_info = get_dataset_names_from_db()    
plot()

