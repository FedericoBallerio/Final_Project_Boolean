# **LOADING AND CLEANING**

# Loading libraries and data from Kaggle
import numpy as np
import pandas as pd
import kaggle
import xlrd
from matplotlib import pyplot as plt
import seaborn as sns
import os
# os.makedirs('./data')     --->       remove # the first time you play the code to create the folder
os.chdir('./data')
kaggle.api.authenticate()
kaggle.api.dataset_download_files('yesshivam007/superstore-dataset',  unzip=True)

# Checking directory contents and loading main dataset
print(os.listdir())

P = pd.read_excel('Sample - Superstore.xls')
P.head()

# Column manipulation and creation of useful values for subsequent predictive model
P.drop(['Row ID'], axis=1, inplace=True)
P['Order Date'] = pd.to_datetime(P['Order Date'], format='%d-%m-%Y')
P['Order Date_month'] = P['Order Date'].dt.month
P['Order Date_year'] = P['Order Date'].dt.year
P.info()

# Loading returns data and merging with main data
R = pd.read_excel('Sample - Superstore.xls' , sheet_name='Returns')
R.head()

df = P.merge(R, on='Order ID', how='left')
df.head()

# Checking for duplicates and null values
print("Duplicate values: ", df.duplicated().sum())
print('\n', 
      "Null values: ",'\n', df.isnull().sum())

# Removing duplicate records and fixing 'Returned' column
dupl = df[df.duplicated()].index
df.drop(dupl, axis=0, inplace=True)

df['Returned'] = df['Returned'].replace('Yes', 1).fillna(0).astype(int)
df.info()

# Exporting cleaned data for Tableau and return to the previous directory 
df.to_excel('df_tableau.xlsx')

os.chdir ('..')


# **Visual analysis on Tableau and development of relevant insights**


# **ADVANCED ANALYSIS**

## **Permutation test**
# Permutation test to statistically validate if discount levels impact profit 
# Simulates 10,000 random reassignments of profit values to evaluate if observed difference between groups occurs by chance
# Visualizes distribution of simulated differences with significance threshold at 5%

# Performing t-test to compare profits with and without discounts
from scipy import stats
stats.ttest_ind(df['Profit'],\
                df[df['Discount'] <= 0.2]['Profit'],\
                alternative='less')

# Permutation test function to validate discount impact on profits
def permutation_test(df, n, plot=True):
    prf = df['Profit']
    prf_30 = df[df['Discount'] <= 0.2]['Profit']
    lnt = len(prf)
    obs_diff = prf.mean() - prf_30.mean()
    comb = np.concatenate([prf, prf_30])
    diffs = []
    for _ in range(n):
        new = np.random.permutation(comb)
        diff_ = new[:lnt].mean() - new[lnt:].mean()
        diffs.append(diff_)
    if plot:
      plt.figure(figsize=(10, 6))
      sns.histplot(diffs, bins=50)
      plt.axvline(obs_diff, color='r', linestyle='--', 
                  label=f'Observed difference: {obs_diff:.4f}')
      plt.axvline(np.percentile(diffs, 5), color='y', linestyle='--', 
                  label=f'Alpha (5%): {np.percentile(diffs, 95):.4f}')
      plt.title('Distribution of simulated differences under H0')
      plt.xlabel('Differences between means')
      plt.ylabel('Frequency')
      plt.legend()
      plt.show()
    p_value = np.mean([abs(d) >= abs(obs_diff) for d in diffs])
    return obs_diff, diffs, p_value

# Running permutation test with 10,000 iterations
obs_diff, diffs, p_value = permutation_test(df, 10000)
print(f"Observed difference: {obs_diff:.4f}")
print(f"p-value: {p_value:.4f}")

# Statistical interpretation of permutation test results
# Noting that the p-value is below the 5% significance level, this result is statistically significant
# This means that by removing discounts greater than 30%, there would be a significant non-random increase in profits


## **Logistic Regression**
# Logistic Regression to predict product returns 
# Due to dataset imbalance, applying resampling techniques with a 1.75:1 ratio of non-returned to returned items
# This ensures the model can effectively learn return patterns without majority class bias

# Basic statistical description of the dataset
df.describe()

# Import necessary machine learning libraries 
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Initialize empty lists to track model performance metrics on both training and test sets
accuracies_Train = []
precisions_Train = []
recalls_Train = []
f1_Train = []
confusion_matrices_Train = []

accuracies_Test = []
precisions_Test = []
recalls_Test = []
f1_Test = []
confusion_matrices_Test = []

# Define sample size for balanced dataset 
len_retr = len(df[df['Returned'] == 1])*1.75

# Create dataset copy, remove irrelevant features, and check remaining structure
pre_mod = df.copy().drop([
    'Order ID', 'Order Date', 'Ship Date', 'Customer Name', 'Country', 
    'Postal Code', 'Segment', 'Product ID', 'Product Name', 'State', 
    'Category', 'Sub-Category', 'Profit', 'Quantity', 'Discount', 
    'Region','Order Date_year'], axis=1)
pre_mod.info()

# Convert categorical variables to binary indicator columns and count resulting features
pre_mod = pd.get_dummies(pre_mod, columns=(['Order Date_month']), dtype=int)
pre_mod = pd.get_dummies(pre_mod, dtype=int)
len(pre_mod.columns)

# Running logistic regression model 100 times with different samplings to ensure result validity
# This provides variety in training data and produces a more reliable model that isn't dependent on a single random sample
for i in range(100):
    
    df_mod = pd.concat([
        pre_mod[pre_mod['Returned'] == 0].sample(n=int(len_retr), replace=False),
        pre_mod[pre_mod['Returned'] == 1]])
    
    y = df_mod['Returned']
    X = df_mod.drop(['Returned'], axis=1)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    log_reg = LogisticRegression(max_iter=5000, solver='liblinear')
    log_reg.fit(X_train, y_train)
    pred_train = log_reg.predict(X_train)
    pred_test = log_reg.predict(X_test)
    accuracies_Train.append(metrics.accuracy_score(y_train, pred_train))
    precisions_Train.append(metrics.precision_score(y_train, pred_train))
    recalls_Train.append(metrics.recall_score(y_train, pred_train))
    f1_Train.append(metrics.f1_score(y_train, pred_train))
    confusion_matrices_Train.append(metrics.confusion_matrix(y_train, pred_train))
    accuracies_Test.append(metrics.accuracy_score(y_test, pred_test))
    precisions_Test.append(metrics.precision_score(y_test, pred_test))
    recalls_Test.append(metrics.recall_score(y_test, pred_test))
    f1_Test.append(metrics.f1_score(y_test, pred_test))
    confusion_matrices_Test.append(metrics.confusion_matrix(y_test, pred_test))

# Calculates and prints classification metrics for training and test datasets    
print(f"Classification Metrics Train:\n"
    f"Average Accuracy: {np.mean(accuracies_Train)*100:.2f}%\n"
    f"Average Precision: {np.mean(precisions_Train)*100:.2f}%\n"
    f"Average Recall: {np.mean(recalls_Train)*100:.2f}%\n"
    f"Average F1 Score: {np.mean(f1_Train)*100:.2f}%\n"
    f"Average Confusion Matrix: {np.mean(confusion_matrices_Train, axis=0)}\n\n"
    f"Classification Metrics Test:\n"
    f"Average Accuracy: {np.mean(accuracies_Test)*100:.2f}%\n"
    f"Average Precision: {np.mean(precisions_Test)*100:.2f}%\n"
    f"Average Recall: {np.mean(recalls_Test)*100:.2f}%\n"
    f"Average F1 Score: {np.mean(f1_Test)*100:.2f}%\n"
    f"Average Confusion Matrix: {np.mean(confusion_matrices_Test, axis=0)}")