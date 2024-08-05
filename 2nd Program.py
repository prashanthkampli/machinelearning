import numpy as np
import pandas as pd

data = pd.read_csv("candidate.csv")
concepts = np.array(data.iloc[:, :-1])
target = np.array(data.iloc[:, -1])

def learn(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [['?' for _ in range(len(specific_h))] for _ in range(len(specific_h))]

    for i, h in enumerate(concepts):
        if target[i] == "Yes":
            specific_h = [s if s == x else '?' for s, x in zip(specific_h, h)]
            general_h = [[g if g == '?' else s for g, s in zip(row, specific_h)] for row in general_h]
        elif target[i] == "No":
            for j in range(len(specific_h)):
                if h[j] != specific_h[j]:
                    general_h[j][j] = specific_h[j]
                else:
                    general_h[j][j] = '?'
        
        print(f"Step {i + 1} of Candidate Elimination Algorithm")
        print("Specific_h:", specific_h)
        print("General_h:", general_h)

    general_h = [row for row in general_h if any(v != '?' for v in row)]
    
    return specific_h, general_h

s_final, g_final = learn(concepts, target)
print("Final Specific_h:", s_final, sep="\n")
print("Final General_h:", g_final, sep="\n")
