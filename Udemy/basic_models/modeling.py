import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#---------------------------------------------------------- Linear Regression ----------------------------------------------------------
# EDA
df = USAhousing
USAhousing.info()
USAhousing.describe()
sns.pairplot(USAhousing) # pairplot showing relationships
sns.distplot(USAhousing['Price']) # displot of column
sns.heatmap(USAhousing.corr())

# Preparing data for regression model
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

lm = LinearRegression()
lm.fit(X_train,y_train)

# Model coefficients for features
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df

# Predictions for test data
predictions = lm.predict(X_test)

# Predictions vs labels scatter
plt.scatter(y_test,predictions)

#Distplot of residuals
sns.distplot((y_test-predictions),bins=50);

# Error Metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#---------------------------------------------------------- Logistic Regression ----------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


train = pd.read_csv('titanic_train.csv')

# EDA
# Plot of nulls if applicable
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')

# Getting dummies for categorical data
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)

# Drop original columns after dummies
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)

# Logistic Model
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

# Predictions
predictions = logmodel.predict(X_test)

#Precision, recall, f1-score, support
print(classification_report(y_test,predictions))

#---------------------------------------------------------- K-Nearest Neighbors ----------------------------------------------------------

#1 Standardize data 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix


scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))


X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],
                                                    test_size=0.30)

                                                 
knn = KNeighborsClassifier(n_neighbors=1) # Initiate model
knn.fit(X_train,y_train) # fit to data
pred = knn.predict(X_test) # Predict

# Results
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

#---------------------------------------------------------- Decision Trees/RF ----------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Pairplot to look at feature relationships
sns.pairplot(df,hue='Kyphosis',palette='Set1')

X = df.drop('Kyphosis',axis=1) #drop target from feature set
y = df['Kyphosis']

# DECISION TREE
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

# Visualizing a tree result
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydo

features = list(df.columns[1:])

dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())  

# RANDOM FOREST
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))

#---------------------------------------------------------- Gradient Boosted Model ----------------------------------------------------------
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

y_train = train_data["Survived"]
train_data.drop(labels="Survived", axis=1, inplace=True)
full_data = train_data.append(test_data)

drop_columns = ["Name", "Age", "SibSp", "Ticket", "Cabin", "Parch", "Embarked"]
full_data.drop(labels=drop_columns, axis=1, inplace=True)

full_data = pd.get_dummies(full_data, columns=["Sex"])
full_data.fillna(value=0.0, inplace=True)

X_train = full_data.values[0:891]
X_test = full_data.values[891:]

# Scale the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

state = 12  
test_size = 0.30  
  
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,  
    test_size=test_size, random_state=state)


# Varying the learning rate (various models)
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    gb_clf.fit(X_train, y_train)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_val, y_val)))


# Final Model
gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
gb_clf2.fit(X_train, y_train)
predictions = gb_clf2.predict(X_val)

print("Confusion Matrix:")
print(confusion_matrix(y_val, predictions))

print("Classification Report")
print(classification_report(y_val, predictions))


# XGBoost Model
xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train)
score = xgb_clf.score(X_val, y_val)
print(score)


#---------------------------------------------------------- K Means ----------------------------------------------------------
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Create Data
data = make_blobs(n_samples=200, n_features=2, 
                           centers=4, cluster_std=1.8,random_state=101)


plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')

# Creating clusters
kmeans = KMeans(n_clusters=4)
kmeans.fit(data[0])

# Cluster center:
kmeans.cluster_centers_

# Labels
kmeans.labels_

# Vizualizing Clusters
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')