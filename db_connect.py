import mysql.connector

mydb = None

try:
    mydb = mysql.connector.connect(

    host="localhost",
    user="root",
    passwd="",
    database="pykater"
    
    )

except Exception as e:
    print(e)
    print("Could not connect to database")    
