import mysql.connector

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

def insert_results(val):

      #val = [
      # ('', 'test_method', 'test params', 10, 'test datset', 0.567, 0.2)
      #]
      
      mycursor = mydb.cursor()
      
      sql = "INSERT INTO results (preprocessings, method, parameters, cv, dataset, accuracy, std_dev) VALUES (%s, %s, %s, %s, %s, %s, %s)"
           
      mycursor.executemany(sql, val)
      
      mydb.commit()
      
      print("DATABASE:", mycursor.rowcount, "results were inserted into table 'results'.")

