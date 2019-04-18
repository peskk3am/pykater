import mysql.connector

mydb = None

try:
    mydb = mysql.connector.connect(
    
      host="www-lab.ms.mff.cuni.cz",
      user="peskk3am",
      passwd="xxx",
      database="peskk3am"
    
    #  host="uvds124.active24.cz",
    #  user="addressc_klara",
    #  passwd="xxx",
    #  database="addressc_klara"
    )
except:
    print("Could not connect to database")    


def insert_results(val):

    #val = [
    # (1, 'test_search', '', 'test_method', 'test params', 10, 'test datset', 0.567, 0.2)
    #]
    
    if mydb:  
        mycursor = mydb.cursor()
          
        sql = "INSERT INTO results (experiment_id, search, preprocessings, method, parameters, cv, dataset, accuracy, std_dev) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
               
        mycursor.executemany(sql, val)
          
        mydb.commit()
          
        print("DATABASE:", mycursor.rowcount, "results were inserted into table 'results'.")


def get_experiment_id():
    
    if mydb:
        mycursor = mydb.cursor()
        
        mycursor.execute("SELECT MAX(experiment_id) FROM results")
        
        myresult = mycursor.fetchall()
        
        try:
            for x in myresult:
                return int(x[0])+1
        except:
            pass
        
        return 0    
   