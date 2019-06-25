
m = "AdaBoostClassifier"
d = 12

f = open(m+"_"+str(d)+".txt")
lines = f.readlines()
f.close()

unique_lines = set()

for line in lines[1:]:
    unique_lines.add(line)
    
f = open(m+"_"+str(d)+"_clean.txt", "w")
f.write(lines[0])
for ul in unique_lines:
    f.write(ul)

f.close()    

