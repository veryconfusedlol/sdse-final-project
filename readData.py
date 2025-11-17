# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




# Reading in the data
data = pd.read_csv("car_data.csv")

# 
print(data.head(5))


categorical = ['type', 'drive', 'make', 'model','transmission']
numerical = ['cylinders', 'displacement']
output = 'combination_mpg' 

num_unique = []
names_unique = {}
# how many classes in each categorical parameter?
for col in categorical:
    print(f"\n{col}: {data[col].nunique()} unique classes")
    num_unique.append( data[col].nunique())
    names_unique[col] = data[col].unique()
    
    # plotting histograms
    plt.figure(figsize=(8, 4))
    data[col].value_counts().plot(kind='bar')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")


plt.show()  # single show after the loop





