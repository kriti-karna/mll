import numpy as np
import pandas as pd

data = pd.read_csv("Enjoy-sport.csv")
concepts = np.array(data)[:,:-1]
print(concepts,"\n")
target = np.array(data)[:,-1]
print(target,"\n")
def learn(c, t):
    specific_h = c[0].copy()
    ln = len(specific_h)
    print("specific_h and general_h:")
    print(specific_h,"\n")
    general_h = [["?" for i in range(ln)] for i in range(ln)]

    for i, h in enumerate(c):
        if t[i] == "yes":
            for x in range(ln):
                if h[x]!= specific_h[x]:
                    specific_h[x] ='?'
                    general_h[x][x] ='?'

        print(specific_h,"\n")
        if t[i] == "no":
            for x in range(ln):
                if h[x]!= specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])
    return specific_h, general_h
s_final, g_final = learn(concepts, target)
print("Final Specific_h:", s_final)
print("Final General_h:", g_final)
data.head()
