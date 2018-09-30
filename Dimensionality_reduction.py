#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 08:18:03 2018

@author: huangsida
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

X, y = df_wine.iloc[:, 2:].values, df_wine.iloc[:, 0].values

from sklearn.ensemble import RandomForestClassifier

# Isolate Data, class labels and column value
names = df_wine.columns[1:].values
# Build the model
rfc = RandomForestClassifier()

# Fit the model
rfc.fit(X, y)

# Print the results
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), rfc.feature_importances_), names), reverse=True))

# Isolate feature importances 
importance = rfc.feature_importances_

# Sort the feature importances 
sorted_importances = np.argsort(importance)

# Insert padding
padding = np.arange(len(names)-1) + 0.5

# Plot the data
plt.barh(padding, importance[sorted_importances], align='center')

# Customize the plot
plt.yticks(padding, names[sorted_importances])
plt.xlabel("Relative Importance")
plt.title("Variable Importance")
plt.savefig('images/10_01.png', dpi=300)
plt.show()

cols = ['Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium']
sns.pairplot(df_wine[cols], size=2.5)
plt.tight_layout()
plt.savefig('images/10_03.png', dpi=300)
plt.show()

cm = np.corrcoef(df_wine[cols].values.T)
#sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)

plt.tight_layout()
plt.savefig('images/10_04.png', dpi=300)
plt.show()

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                     stratify=y,
                     random_state=42)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

print('Baseline')
lr = LogisticRegression()
lr = lr.fit(X_train_std, y_train)

y_pred = lr.predict(X_train_std)
print('Accuracy_lr_train: %.2f' % accuracy_score(y_train, y_pred))

y_pred = lr.predict(X_test_std)
print('Accuracy_lr_test: %.2f' % accuracy_score(y_test, y_pred))

svm = SVC(kernel='rbf', random_state=1, gamma=0.1, C=1.0)
svm.fit(X_train_std, y_train)

y_pred = svm.predict(X_train_std)
print('Accuracy_svm_train: %.2f' % accuracy_score(y_train, y_pred))


y_pred = svm.predict(X_test_std)
print('Accuracy_svm_test: %.2f\n' % accuracy_score(y_test, y_pred))

print('PCA')
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

lr = LogisticRegression()
lr = lr.fit(X_train_pca, y_train)

y_pred = lr.predict(X_train_pca)
print('Accuracy_lr_train: %.2f' % accuracy_score(y_train, y_pred))

y_pred = lr.predict(X_test_pca)
print('Accuracy_lr_test: %.2f' % accuracy_score(y_test, y_pred))

svm = SVC(kernel='rbf', random_state=1, gamma=0.1, C=1.0)
svm.fit(X_train_pca, y_train)

y_pred = svm.predict(X_train_pca)
print('Accuracy_svm_train: %.2f' % accuracy_score(y_train, y_pred))


y_pred = svm.predict(X_test_pca)
print('Accuracy_svm_test: %.2f\n' % accuracy_score(y_test, y_pred))


print('LDA')
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)

y_pred = lr.predict(X_train_lda)
print('Accuracy_lr_train: %.2f' % accuracy_score(y_train, y_pred))

y_pred = lr.predict(X_test_lda)
print('Accuracy_lr_test: %.2f' % accuracy_score(y_test, y_pred))

svm = SVC(kernel='rbf', random_state=1, gamma=0.1, C=1.0)
svm.fit(X_train_lda, y_train)

y_pred=svm.predict(X_train_lda)
print('Accuracy_svm_train: %.2f' % accuracy_score(y_train, y_pred))

y_pred = svm.predict(X_test_lda)
print('Accuracy_svm_test: %.2f\n' % accuracy_score(y_test, y_pred))

print('kPCA')
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.5)
X_train_skernpca = scikit_kpca.fit_transform(X_train_std,y_train)
X_test_skernpca = scikit_kpca.transform(X_test_std)

lr = LogisticRegression()
lr = lr.fit(X_train_skernpca, y_train)

y_pred = lr.predict(X_train_skernpca)
print('Accuracy_lr_train: %.2f' % accuracy_score(y_train, y_pred))

y_pred = lr.predict(X_test_skernpca)
print('Accuracy_lr_test: %.2f' % accuracy_score(y_test, y_pred))

svm = SVC(kernel='rbf', random_state=1, gamma=0.1, C=1.0)
svm.fit(X_train_skernpca, y_train)

y_pred=svm.predict(X_train_skernpca)
print('Accuracy_svm_train: %.2f' % accuracy_score(y_train, y_pred))

y_pred = svm.predict(X_test_skernpca)
print('Accuracy_svm_test: %.2f\n' % accuracy_score(y_test, y_pred))

print("My name is Huang Sida")
print("My NetID is sidah2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")




