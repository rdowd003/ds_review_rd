# Scaling features

import pandas as pd
import numpy as np
import sklearn.preprocessing as preproc

df = pd.read_csv('df_filename.csv')

# log transform
df['log-feature'] = np.log10(df['feature'])

# Min-max scaling (0-1)
df['minmax_feature'] = preproc.minmax_scale(df[['feature']])

# Stadnardization (variance scaling, mean = 0, var = 1)
df['standardized'] = preproc.StandardSCaler().fit_transform(df['feature_column'])

# L2 Norm
df['l2_norm'] = preproc.normalize(df['feature'],axis=0)