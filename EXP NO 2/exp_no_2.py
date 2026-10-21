
import numpy as np
import pandas as pd
import os

data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'enjoysport.csv'))

concepts = np.array(data.iloc[:, 0:-1])
print(concepts)

target = np.array(data.iloc[:, -1])
print(target)


def learn(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [["?" for i in range(len(specific_h))]
                 for i in range(len(specific_h))]

    print("Initial Specific Hypothesis:")
    print(specific_h)

    print("Initial General Hypothesis:")
    print(general_h)

    for i, h in enumerate(concepts):
        if target[i] == "yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'

        elif target[i] == "no":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

        print("Step", i + 1)
        print("Specific Hypothesis:", specific_h)
        print("General Hypothesis:", general_h)

    indices = [
        i for i, val in enumerate(general_h)
        if val == ['?'] * len(specific_h)
    ]

    for i in reversed(indices):
        general_h.pop(i)

    return specific_h, general_h


s_final, g_final = learn(concepts, target)

print("Final Specific Hypothesis:", s_final)
print("Final General Hypothesis:", g_final)
