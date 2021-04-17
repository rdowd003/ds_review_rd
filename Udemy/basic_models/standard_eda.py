import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# pair plot
sns.pairplot(df,hue='Kyphosis',palette='Set1') # Hue = target good for Survivor 

# Distplot
sns.distplot(df['column'])

# Scatter plot
plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')

# Heat map of nulls
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')