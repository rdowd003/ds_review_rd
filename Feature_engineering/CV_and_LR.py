import pandas as import pd
import numpy as np
import json

from sklearn import linear_model,metrics
from sklearn.model_selection import cross_val_score, test_train_split

# Open .json file

df_file = open('filename.json')
df = pd.DataFrame([json.loads(x) for x in filename.readlines()])
df_file.close()

# Split data if necess
X_train, X_test, y_train, y_test = train_test_split(df, df.target, test_size=0.4, random_state=0)

df['log_counts'] = np.log10(df['counts']+1)
lr_model = linear_model.LinearRegression()
# scores = cross_val_score(model, df_features, df.target, cv=# cv's, scoring = 'f1_macro)
scores = cross_val_score(lr_model, df[['log_count']],df['target'],cv=10) ## array of scores as outpout, 1 value per model