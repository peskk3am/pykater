import mysql.connector
from db_connect import *
import _config

import numpy

def my_isnan(number):
    res = False
    try:
        res = numpy.isnan(number)
    except:
        pass
    
    return res    


def insert_results(val):

    #val = [
    # (experiment_id, search, pre, method, params, cv, dataset, accuracy, std_dev, generation)
    #]
    new_val = []
    for val_tuple in val:      
        val_list = list(val_tuple)
        val_list = [None if my_isnan(v) else v for v in val_list]                
        new_val.append(tuple(val_list))
                    
    val = new_val   
    # print(val)
    print()
        
    if mydb and _config.save_results_to_db:  
        mycursor = mydb.cursor()
          
        sql = "INSERT INTO results (experiment_id, search, preprocessings, method, parameters, cv, dataset, accuracy, std_dev, generation) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
               
        mycursor.executemany(sql, val)
          
        mydb.commit()
          
        print("DATABASE:", mycursor.rowcount, "results were inserted into table 'results'.")


def get_experiment_id():
    
    if mydb and _config.save_results_to_db:
        mycursor = mydb.cursor()
        
        mycursor.execute("SELECT MAX(experiment_id) FROM results")
        
        myresult = mycursor.fetchall()
        
        try:
            for x in myresult:
                return int(x[0])+1
        except:
            pass
        
        return 0    


def insert_datasets(val):  
    #  val: [(name, openml_id, instances, attributes, classes, task)]
        
    if mydb and _config.save_results_to_db:  
        mycursor = mydb.cursor()
        print(val)  
        sql = "INSERT IGNORE INTO datasets (name, openml_id, instances, attributes, classes, task) VALUES (%s, %s, %s, %s, %s, %s)"
               
        mycursor.executemany(sql, val)
          
        mydb.commit()
          
        print("DATABASE:", mycursor.rowcount, "dataset was inserted into table 'datasets'.")
   