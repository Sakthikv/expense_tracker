import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('data.csv')

# Extract concepts and target
concepts = np.array(data)[:,:-1]
target = np.array(data)[:,-1]

# Function to train and find the hypothesis
def train(con, tar):
    for i, val in enumerate(tar):
        if val == 'yes':
            specific_h = con[i].copy()
            break
    for i, val in enumerate(con):
        if tar[i] == 'yes':
            for x in range(len(specific_h)):
                if val[x] != specific_h[x]:
                    specific_h[x] = '?'
                else:
                    pass
    return specific_h

# Train the model and print the hypothesis
print("Final Specific Hypothesis:\n")
print(train(concepts, target))