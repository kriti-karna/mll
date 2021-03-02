import numpy as np
import csv

with open("finds.csv",'r') as f:
    reader = csv.reader(f)
    data = list(reader)

print(data)

d = np.array(data)[:,:-1]
print("Attributes:\n",d)

h = ['0', '0', '0', '0', '0', '0']

for row in data:
    if row[-1] == "yes":
        j=0
        for col in row:
            if col!="yes":
                if col!=h[j] and h[j]=='0':
                    h[j] = col
                elif col!=h[j] and h[j]!='0':
                    h[j] = '?'
            j=j+1

print("final hypothesis: ",h)
