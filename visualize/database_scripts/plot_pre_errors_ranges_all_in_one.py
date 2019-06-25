import sys
import os

from db_connect import *

import matplotlib.pyplot as plt
import numpy 
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
    43,
    313,
    18,
    311,
    12,
    16,                
]

labels_indexes = []
datasets_info = {}

def my_sort(data, data_pre):
    # sort by (max - min)
    global labels_indexes
    
    criteria = []  # [(datasets_index, max-min)]
    stats = {}  # [(openml_id, min, max, mean, 25percentil, 75percentil, max-min)]
    stats_pre = {}
    
    for i in range(len(data)):   #i ... index in datasets and in data       
       if len(data[i]) == 0:
           continue
       d = data[i]
       d_pre = data_pre[i]    
       criteria.append((i, max(d) - min(d)))       
       
       if print_all_stats or print_all_stats_tex_table:
           box = percentile(d,75)-percentile(d,25)
           box_pre = percentile(d_pre,75)-percentile(d_pre,25)
           stats[datasets[i]] = (round(min(d),2), round(percentile(d,25),2), round(median(d),2), round(percentile(d,75),2), round(max(d),2), round(box,2))
           stats_pre[datasets[i]] = (round(min(d_pre),2), round(percentile(d_pre,25),2), round(median(d_pre),2), round(percentile(d_pre,75),2), round(max(d_pre),2), round(box_pre,2))       
       
    criteria.sort(key=lambda x: x[1])
        
    data_sorted = []
    data_pre_sorted = []
    for c in criteria:
        print(c)
        data_sorted.append(data[c[0]])
        data_pre_sorted.append(data_pre[c[0]])
        labels_indexes.append(c[0])

    
    if print_all_stats or print_all_stats_tex_table:
        stats_sorted = []
        stats_pre_sorted = []
        for c in criteria:        
            stats_sorted.append(stats[datasets[c[0]]])
            stats_pre_sorted.append(stats_pre[datasets[c[0]]])        
            

    if print_all_stats:
        f = open("PRE_all_stats.txt", 'w')
        for s in stats_sorted:
            f.write(str(s)+"\n")
        f.close()
    
        f = open("PRE_all_stats_pre.txt", 'w')
        for s in stats_pre_sorted:
            f.write(str(s)+"\n")
        f.close()


    if print_all_stats_tex_table:
        f = open("PRE_all_stats_tex_table.txt", 'w')
        # write header
        f.write("Task (Inst, Attrs, Classes) & Min & 25\\textsuperscript{th} Perc. & Median & 75\\textsuperscript{th} Perc. & Max & Box Size\\\\\n")
        f.write("\\hline\\hline\n")
        for index in labels_indexes:        
            name, instances, attributes, classes = datasets_info.get(datasets[index], ("???", "???", "???", "???"))
            f.write(name+" ("+str(instances)+", "+str(attributes)+", "+str(classes)+") &")
                    
            s = stats[datasets[index]]
            values_str = str(s).strip("()").replace(", ", "&")
            f.write(values_str+"\\\\\n")
    
            # draw a line
            #f.write("\\hline\n")
            
            # write pre line        
            f.write("~~preprocessings &")
                    
            s = stats_pre[datasets[index]]
            values_str = str(s).strip("()").replace(", ", "&")
            f.write(values_str+"\\\\\n")
            
            # draw a line
            f.write("\\hline\n")
        f.close()

      
    f = open("labels_indexes_pre.txt", 'w')
    f.write(str(labels_indexes))
    f.close()
    
    return data_sorted, data_pre_sorted      


def plot():
        
    # data ... sequence of vectors
    data = []
    data_pre = []
    for d in datasets:
        d, d_pre = get_data_from_db(d)
        data.append(d)
        data_pre.append(d_pre)
    
    #labels_indexes = range(len(datasets))  # when we don't sort
    data, data_pre = my_sort(data, data_pre)        
    
    # create labels
    labels = []
    for i in labels_indexes:
        name, instances, attributes, classes = datasets_info.get(datasets[i], ("???", "???", "???", "???"))
        l = str(datasets[i])+" "+name+" ("+str(instances)+", "+str(attributes)+", "+str(classes)+")" 
        labels.append(l)         
        
        
    fig1, ax1 = plt.subplots(figsize=(12, 9))    
    ax1.set_title("Errors ranges - methods vs. methods with preprocessings")
    
    offset = -0.2
    pos = numpy.arange(len(data))+offset
    ax1.boxplot(data, positions=pos, widths=0.2, manage_xticks=False)    
    
    offset = 0.2
    pos = numpy.arange(len(data))+offset
    bp_pre = ax1.boxplot(data_pre, positions=pos, widths=0.2,  manage_xticks=False)
    
    color = "xkcd:clear blue"
    for element in ['boxes', 'whiskers', 'fliers', 'caps']:    
        plt.setp(bp_pre[element], color=color)    
    plt.setp(bp_pre["fliers"], markeredgecolor=color)

    
    ax1.set_ylabel('Error')
    print(len(labels), len(data))
    print(labels)
    
    plt.xticks(range(len(data)), labels, rotation='vertical')
    plt.subplots_adjust(bottom=0.35)
    
    plt.savefig('all_pre.png')
    plt.savefig('all_pre.pdf')
    plt.close(fig1)    
        

def get_data_from_db(dataset):
    
    if mydb:
        mycursor = mydb.cursor()                                          
        
        val = (dataset,)
        
        sql = "SELECT accuracy, preprocessings FROM results WHERE dataset=%s AND accuracy IS NOT NULL AND preprocessings='[]'"
        mycursor.execute(sql, val)        
        
        myresult = mycursor.fetchall()
        
        errors = []        
        try:
            for x in myresult:
                errors.append( 1-float(x[0]) )
                if x[1] != "[]":
                    print(x[1])
        except:
            pass
        
        
        sql = "SELECT accuracy FROM results WHERE dataset=%s AND accuracy IS NOT NULL AND preprocessings!='[]'"
        mycursor.execute(sql, val)
        myresult = mycursor.fetchall()

        errors_pre = []
        try:
            for x in myresult:
                errors_pre.append( 1-float(x[0]) )
        except:
            pass
                
        return errors, errors_pre    


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

