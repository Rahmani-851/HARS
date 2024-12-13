# %%
import subprocess
subprocess.run(["pip", "install", "seaborn"], check=True)
#subprocess.run(["pip", "install", "tensorflow"], check=True)
import argparse
import mlflow
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model, model_selection, preprocessing, metrics
#import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


parser = argparse.ArgumentParser()

parser.add_argument("--trainData", type=str, help="Path to training dataset")
args = parser.parse_args()
mlflow.autolog()


# %%
DSTrain = pd.read_csv(args.trainData)
#DSTrain = pd.read_csv("train.csv")
DSTest = pd.read_csv("test.csv")

# %%
#DSTrain.shape
DSTest.shape

# %%
DSTrain.duplicated().any()

# %%
data = pd.concat([DSTrain, DSTest], axis=0)
data.reset_index(drop = True, inplace = True)

data.isna().sum()

data.iloc[:, :-1].min().value_counts()

data.iloc[:, :-1].max().value_counts()

data.Activity.value_counts()

# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['Activity'] = le.fit_transform(data.Activity)
data['Activity'].sample(5)

# %%
# Calculate the correlation values
feature_cols = data.columns[:-1]
corr_values = data[feature_cols].corr()

# Simplify by emptying all the data below the diagonal
tril_index = np.tril_indices_from(corr_values)

# Make the unused values NaNs
for coord in zip(*tril_index):
    corr_values.iloc[coord[0], coord[1]] = np.NaN
    
# Stack the data and convert to a data frame
corr_values = (corr_values
               .stack()
               .to_frame()
               .reset_index()
               .rename(columns={'level_0':'feature1',
                                'level_1':'feature2',
                                0:'correlation'}))

# Get the absolute values for sorting
corr_values['abs_correlation'] = corr_values.correlation.abs()

# %%
sns.set_context('talk')
sns.set_style('white')

ax = corr_values.abs_correlation.hist(bins=50, figsize=(12, 8))
ax.set(xlabel='Absolute Correlation', ylabel='Frequency');

# %%
# The most highly correlated values
corr_values.sort_values('correlation', ascending=False).query('abs_correlation>0.8')

# %%
from sklearn.model_selection import StratifiedShuffleSplit

# Get the split indexes
strat_shuf_split = StratifiedShuffleSplit(n_splits=1, 
                                          test_size=0.3, 
                                          random_state=42)

train_idx, test_idx = next(strat_shuf_split.split(data[feature_cols], data.Activity))

# Create the dataframes
X_train = data.loc[train_idx, feature_cols]
y_train = data.loc[train_idx, 'Activity']

X_test  = data.loc[test_idx, feature_cols]
y_test  = data.loc[test_idx, 'Activity']

# %%
y_train.value_counts(normalize=True)

# %%
y_test.value_counts(normalize=True)

# %%
from sklearn.linear_model import LogisticRegression

# Standard logistic regression
lr = LogisticRegression(solver='liblinear').fit(X_train, y_train)

# %%
from sklearn.linear_model import LogisticRegressionCV

# L1 regularized logistic regression
lr_l1 = LogisticRegressionCV(Cs=10, cv=4, penalty='l1', solver='liblinear').fit(X_train, y_train)

# %%
# L2 regularized logistic regression
lr_l2 = LogisticRegressionCV(Cs=10, cv=4, penalty='l2', solver='liblinear').fit(X_train, y_train)

# %%
coefficients = list()

coeff_labels = ['lr', 'l1', 'l2']
coeff_models = [lr, lr_l1, lr_l2]

for lab,mod in zip(coeff_labels, coeff_models):
    coeffs = mod.coef_
    coeff_label = pd.MultiIndex(levels=[[lab], [0,1,2,3,4,5]], 
                                 codes=[[0,0,0,0,0,0], [0,1,2,3,4,5]])
    coefficients.append(pd.DataFrame(coeffs.T, columns=coeff_label))

coefficients = pd.concat(coefficients, axis=1)

coefficients.sample(10)

# %%
fig, axList = plt.subplots(nrows=3, ncols=2)
axList = axList.flatten()
fig.set_size_inches(10,10)

for ax in enumerate(axList):
    loc = ax[0]
    ax = ax[1]
    
    data = coefficients.xs(loc, level=1, axis=1)
    data.plot(marker='o', ls='', ms=2.0, ax=ax, legend=False)
    
    if ax is axList[0]:
        ax.legend(loc=4)
        
    ax.set(title='Coefficient Set '+str(loc))

plt.tight_layout()

# %%
# Predict the class and the probability for each
y_pred = list()
y_prob = list()

coeff_labels = ['lr', 'l1', 'l2']
coeff_models = [lr, lr_l1, lr_l2]

for lab,mod in zip(coeff_labels, coeff_models):
    y_pred.append(pd.Series(mod.predict(X_test), name=lab))
    y_prob.append(pd.Series(mod.predict_proba(X_test).max(axis=1), name=lab))
    
y_pred = pd.concat(y_pred, axis=1)
y_prob = pd.concat(y_prob, axis=1)

y_pred.head()

# %%
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

metrics = list()
cm = dict()

for lab in coeff_labels:

    # Preciision, recall, f-score from the multi-class support function
    precision, recall, fscore, _ = score(y_test, y_pred[lab], average='weighted')
    
    # The usual way to calculate accuracy
    accuracy = accuracy_score(y_test, y_pred[lab])
    
    # ROC-AUC scores can be calculated by binarizing the data
    auc = roc_auc_score(label_binarize(y_test, classes=[0,1,2,3,4,5]),
              label_binarize(y_pred[lab], classes=[0,1,2,3,4,5]), 
              average='weighted')
    

# Last, the confusion matrix
    cm[lab] = confusion_matrix(y_test, y_pred[lab])
    
    metrics.append(pd.Series({'precision':precision, 'recall':recall, 
                              'fscore':fscore, 'accuracy':accuracy,
                              'auc':auc}, 
                             name=lab))

metrics = pd.concat(metrics, axis=1)

# %%
fig, axList = plt.subplots(nrows=2, ncols=2)
axList = axList.flatten()
fig.set_size_inches(12, 10)

axList[-1].axis('off')

for ax,lab in zip(axList[:-1], coeff_labels):
    sns.heatmap(cm[lab], ax=ax, annot=True, fmt='d');
    ax.set(title=lab);
    
plt.tight_layout()




#Random Forest Integration

from sklearn.ensemble import RandomForestClassifier

# Train a Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=None, class_weight='balanced').fit(X_train, y_train)

# %% Evaluate Random Forest Model
# Predict class labels and probabilities
y_pred_rf = pd.Series(rf.predict(X_test), name='rf')
y_prob_rf = pd.Series(rf.predict_proba(X_test).max(axis=1), name='rf')

# Add predictions to existing DataFrames
y_pred = pd.concat([y_pred, y_pred_rf], axis=1)
y_prob = pd.concat([y_prob, y_prob_rf], axis=1)

# %% Metrics and Confusion Matrix for Random Forest
# Calculate metrics
precision_rf, recall_rf, fscore_rf, _ = score(y_test, y_pred_rf, average='weighted')
accuracy_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(label_binarize(y_test, classes=[0,1,2,3,4,5]),
                       label_binarize(y_pred_rf, classes=[0,1,2,3,4,5]),
                       average='weighted')

# Confusion matrix
cm['rf'] = confusion_matrix(y_test, y_pred_rf)

# Add Random Forest metrics to the DataFrame
metrics['rf'] = pd.Series({'precision': precision_rf, 'recall': recall_rf,
                           'fscore': fscore_rf, 'accuracy': accuracy_rf,
                           'auc': auc_rf})

# %% Plot Confusion Matrix for Random Forest
fig, axList = plt.subplots(nrows=3, ncols=2)
axList = axList.flatten()
fig.set_size_inches(12, 12)

# Add an empty subplot for layout alignment if needed
axList[-1].axis('off')

# Plot all confusion matrices, including Random Forest
for ax, lab in zip(axList[:-1], coeff_labels + ['rf']):
    sns.heatmap(cm[lab], ax=ax, annot=True, fmt='d')
    ax.set(title=lab)

plt.tight_layout()