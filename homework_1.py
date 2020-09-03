#!/usr/bin/env python
# coding: utf-8

# # Homework 1

# # Importing libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.datasets import load_wine
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, datasets
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from matplotlib.colors import ListedColormap
from seaborn import lineplot


# # Loading Wine dataset

# In[3]:


f1 = 'alcohol' 
f2 = 'malic_acid'

dataset = load_wine()
dataframe = pd.DataFrame(dataset.data, columns = dataset.feature_names)
df_2 = dataframe[[f1,f2]]
df_2['label'] = dataset.target
df_2


# # 2D representation 

# In[4]:


c_zero = df_2.loc[df_2['label'] == 0, ['alcohol','malic_acid']]
c_one = df_2.loc[df_2['label'] == 1, ['alcohol','malic_acid']]
c_two = df_2.loc[df_2['label'] == 2, ['alcohol','malic_acid']]

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(c_zero['alcohol'], c_zero['malic_acid'], s=9, color='orange', label = 'wine_0')
ax.scatter(c_one['alcohol'], c_one['malic_acid'], s=9, color='cyan', label = 'wine_1')
ax.scatter(c_two['alcohol'], c_two['malic_acid'], s=9, color='blue', label = 'wine_2')
plt.legend(loc='upper right', fontsize=12)
plt.show()


# In[5]:


# Exploring the cardinality of the classes

len(c_zero), len(c_one), len(c_two)


# # k-NN classification

# In[6]:


# Splitting data in train, validation and test set

X, X_test, y, y_test = train_test_split(df_2[[f1,f2]], df_2['label'], train_size = 0.7, 
                                                    random_state = 42, shuffle = True)

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.7, 
                                                    random_state = 42, shuffle = True)


# # Decision boundaries

# In[7]:


# Function for plotting decision boundaries

def plot_DB(clf, X_train, y_train):
    X_db = X_train.to_numpy()
    x_min, x_max = X_db[:, 0].min() - 1, X_db[:, 0].max() + 1
    y_min, y_max = X_db[:, 1].min() - 1, X_db[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X_db[:, 0], X_db[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('alcohol')
    plt.ylabel('malic_acid')
    plt.show()


# In[8]:


cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

accuracies = {}
K = [1,3,5,7]
for k in K:
    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(X_train, y_train)
    plot_DB(clf, X_train, y_train)
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    accuracies[k] = accuracy
    print(f"Classification report with k = {k}")
    print(classification_report(y_val, y_pred))
accuracies


# In[9]:


# Plotting accuracies on validation set vs k

f, ax = plt.subplots(1,figsize=(10,6))
ax.plot(K, list(accuracies.values()), label = "accuracy", marker='o')
plt.xlabel('K')
plt.ylabel('accuracy score on val')


# In[10]:


# Evaluating the model built with the best k

clf = KNeighborsClassifier(n_neighbors = 7)
clf.fit(X, y)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
accuracy_score(y_test, y_pred)

# Plotting the confusion matrix

cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix, annot=True)


# In[11]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))


# # SVM classification with linear kernel

# In[62]:


C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
accuracies = {}

for c in C:
    clf = SVC(kernel='linear',C=c)
    clf.fit(X_train, y_train)
    plot_DB(clf, X_train, y_train)
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    accuracies[c] = accuracy
    print(f"Classification report with C = {c}")
    print(classification_report(y_val, y_pred))
accuracies


# In[70]:


# Plotting accuracies on validation set vs C 

f, ax = plt.subplots(1,figsize=(10,6))
x_axis = np.arange(len(C))
y_axis = list(accuracies.values())
plt.bar(x_axis,y_axis)
plt.xticks(x, ('C=0.001', 'C=0.01', 'C=0.1', 'C=1', 'C=10', 'C=100', 'C=1000'))
plt.ylabel('accuracy score on val')


# In[71]:


# Evaluating best C on validation set

clf = SVC(C = 1000, kernel='linear')
clf.fit(X, y)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Plotting the confusion matrix

cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix, annot=True)


# # SVM classification with RBF kernel

# In[72]:


C = [0.001, 0.01, 0.1, 1, 10, 100,1000]
accuracies = {}

for c in C:
    clf = SVC(kernel = 'rbf', C=c)
    clf.fit(X_train, y_train)
    plot_DB(clf, X_train, y_train)
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    accuracies[c] = accuracy
    print(f"Classification report with C = {c}")
    print(classification_report(y_val, y_pred))
accuracies


# # Evaluating the best C on the test set

# In[73]:


clf = SVC(C = 10, kernel='rbf')
clf.fit(X, y)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


# # Gamma and C parameter tuning for RBF kernel

# In[76]:


clf = SVC(C = 100, gamma=1e-2, kernel='rbf')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
print(accuracy_score(y_val, y_pred))


# In[77]:


# Grid search

param_grid = {'C': [0.01,0.1,1,10,100], 'gamma': [1e-7,1e-5,1e-3,1e-1,1e-0], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")


# # Training the model with tuned parameters and scoring on val_set

# In[80]:


clf = SVC(C=1, gamma=1e-1, kernel = 'rbf')
clf.fit(X_train, y_train)

plot_DB(clf, X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))


# # KFold

# In[81]:


param_grid = {'C': [0.01,0.1,1,10,100], 'gamma': [1e-7,1e-5,1e-3,1e-1,1e-0], 'kernel': ['rbf']}
cv = KFold(n_splits=5, shuffle=True)
grid = GridSearchCV(SVC(), param_grid, cv=cv)
grid.fit(X, y)
print(f"Best params: {grid.best_params_}")

y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))


# # Extra section

# # Data normalization

# In[85]:


scaler = RobustScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(dataframe), columns=dataframe.columns)
scaled_df['label'] = dataset.target

means_df = pd.DataFrame(columns=scaled_df.columns)
for c in scaled_df.columns:
    l = []
    for label in [0,1,2]:
        l.append(scaled_df.loc[scaled_df['label'] == label, c].mean())
    means_df[c] = l

print("STANDARD DEVIATION OF THE MEANS FOR EACH CLASS LABEL\n")
for c in means_df.columns:
    print(f"std of {c} : {means_df[c].std()}")
    
means_df


# # 2D plot

# In[86]:


f1 = 'flavanoids'
f2 = 'proline'

normdf_2 = scaled_df[[f1,f2]]
normdf_2['label'] = dataset.target

c_zero = normdf_2.loc[df_2['label'] == 0, [f1,f2]]
c_one = normdf_2.loc[df_2['label'] == 1, [f1,f2]]
c_two = normdf_2.loc[df_2['label'] == 2, [f1,f2]]

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(c_zero[f1], c_zero[f2], s=9, color='orange', label = 'wine_0')
ax.scatter(c_one[f1], c_one[f2], s=9, color='cyan', label = 'wine_1')
ax.scatter(c_two[f1], c_two[f2], s=9, color='blue', label = 'wine_2')
plt.legend(loc='upper right', fontsize=12)
ax.set_xlabel("f1")
ax.set_ylabel("f2")
plt.show()


# In[95]:


# Splitting data in train and test set

X, X_test, y, y_test = train_test_split(normdf_2[[f1,f2]], normdf_2['label'], train_size = 0.7, 
                                                    random_state = 42, shuffle = True)


# # k-NN

# In[106]:


param_grid = {'n_neighbors': [3,5,7,9]}
cv = KFold(n_splits=5, shuffle=True)
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv)
grid.fit(X, y)
print(f"Best params: {grid.best_params_}")

y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))


# # SMV with linear kernel

# In[105]:


param_grid = {'C': [0.01,0.1,1,10,100], 'kernel': ['linear']}
cv = KFold(n_splits=5, shuffle=True)
grid = GridSearchCV(SVC(), param_grid, cv=cv)
grid.fit(X, y)
print(f"Best params: {grid.best_params_}")

y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))


# # SMV with rbf kernel

# In[104]:


param_grid = {'C': [0.01,0.1,1,10,100], 'gamma': [1e-7,1e-5,1e-3,1e-1,1e-0], 'kernel': ['rbf']}
cv = KFold(n_splits=5, shuffle=True)
grid = GridSearchCV(SVC(), param_grid, cv=cv)
grid.fit(X, y)
print(f"Best params: {grid.best_params_}")

y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))

