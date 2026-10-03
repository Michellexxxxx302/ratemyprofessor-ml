#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 22:44:24 2025

@author: michellexia
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, permutation_test
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.linear_model import LassoCV



# Load data
num_data_raw = pd.read_csv('rmpCapstoneNum.csv')

# Seed RNG
np.random.seed(19487111)

# Rename columns based on spec sheet
num_data_raw.columns = [
    'Average Rating',
    'Average Difficulty',
    'Number of ratings',
    'Received a pepper?',
    'Proportion who would take again',
    'Number of online ratings',
    'Male gender',
    'Female'
]

# Plot the distribution of nnumber of ratings
filtered_data = num_data_raw[num_data_raw['Number of ratings'] < 70]
plt.figure(figsize=(10, 5))
sns.histplot(filtered_data['Number of ratings'], bins=19, kde=False)
plt.axvline(x=6, color='red', linestyle='--', label='Threshold = 6')
plt.title('Frequency of Number of Ratings')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Professors')
plt.show()

# Take the "Consideration" part into account, filter to only include professors with > 6 ratings
num_data = num_data_raw[num_data_raw['Number of ratings'] > 6].copy()






#%% Q1
#%%
# Filter rows with known gender
gender_data = num_data[(num_data['Male gender'] == 1) | (num_data['Female'] == 1)].copy()

# Mann-Whitney U Test
male_ratings = gender_data[gender_data['Male gender'] == 1]['Average Rating']
female_ratings = gender_data[gender_data['Female'] == 1]['Average Rating']
u_stat, p_val_mwu = mannwhitneyu(male_ratings, female_ratings, alternative='two-sided')

print("Mann-Whitney U Test Results:")
print("U-statistic: ", u_stat)
print("p-value: ", p_val_mwu)

#%%
# Plot the two distribution to see if they have similar distribution pattern
plt.figure(figsize=(8, 5))

plt.hist(
    gender_data[gender_data['Male gender'] == 1]['Average Rating'],
    bins=20, alpha=0.5, label='Male'
)

plt.hist(
    gender_data[gender_data['Female'] == 1]['Average Rating'],
    bins=20, alpha=0.5, label='Female'
)

plt.title("Average Rating Distribution by Gender")
plt.xlabel("Average Rating")
plt.ylabel("Count")
plt.legend()
plt.show()

#%%
# Plot the boxplot the visualize the effect
gender_data['Gender'] = gender_data['Male gender'].apply(lambda x: 'Male')  
gender_data.loc[gender_data['Female'] == 1, 'Gender'] = 'Female'

# Plot boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(data=gender_data, x='Gender', y='Average Rating', palette='pastel')
plt.title('Average Ratings by Gender')
plt.ylabel('Average Rating')
plt.xlabel('Professor Gender')
plt.grid(True)
plt.show()


#%% Q2
#%%
# Drop missing values
valid_data = num_data.dropna(subset=['Average Rating', 'Number of ratings'])
# Spearman correlation test (non-parametric)
corr, p_value = spearmanr(valid_data['Number of ratings'], valid_data['Average Rating'])

print("Spearman correlation:", round(corr, 4))
print("Correlation: ", round(corr,4))
print("p-value: ", p_value)

#%%
# Visualization
plt.figure(figsize=(8, 5))
sns.regplot(x='Number of ratings', y='Average Rating', data=valid_data, scatter_kws={'alpha':0.2}, line_kws={'color':'red'})
plt.title('Average Rating vs. Number of Ratings')
plt.xlabel('Number of Ratings (proxy for experience)')
plt.ylabel('Average Rating')
plt.show()


#%% Q3
#%%
# Drop missing values
df = num_data.dropna(subset=['Average Rating', 'Average Difficulty'])

# Spearman correlation
corr, p_val = spearmanr(df['Average Rating'], df['Average Difficulty'])

print("Spearman correlation: ", round(corr, 4))
print("p-value: ", p_val)

#%%
# Visualization
plt.figure(figsize=(8, 5))
sns.regplot(x='Average Difficulty', y='Average Rating', data=df, scatter_kws={'alpha':0.2}, line_kws={'color':'red'})
plt.title('Average Rating vs. Average Difficulty')
plt.xlabel('Average Difficulty')
plt.ylabel('Average Rating')
plt.show()


#%% Q4
#%%
# Find the right threshold
valid_online_data = num_data.dropna(subset=['Average Rating', 'Number of online ratings'])
online_counts = valid_online_data['Number of online ratings']
plt.figure(figsize=(10, 5))
sns.histplot(online_counts, bins=50, kde=False)
plt.title('Distribution of Number of Online Ratings per Professor')
plt.xlabel('Number of Online Ratings')
plt.ylabel('Number of Professors')
plt.show()

#%%
# Filter data
valid_online_data = num_data.dropna(subset=['Average Rating', 'Number of online ratings'])

online_threshold = 2
online_group = valid_online_data[valid_online_data['Number of online ratings'] >= online_threshold]['Average Rating'].values
offline_group = valid_online_data[valid_online_data['Number of online ratings'] == 0]['Average Rating'].values

def ourTestStatistic(x, y):
    return np.mean(x) - np.mean(y)
dataToUse_Q4 = (online_group, offline_group)
pTest_Q4 = permutation_test(dataToUse_Q4, ourTestStatistic, n_resamples=10000, alternative='two-sided', random_state=19487111)

print('Test statistic:', pTest_Q4.statistic)
print('Exact p-value:', pTest_Q4.pvalue)

#%%
# Plot the two distribution to see if they have similar distribution pattern
plt.figure(figsize=(8, 5))
plt.hist(online_group, bins=20, alpha=0.5, label='Online-heavy (>=2 ratings)')
plt.hist(offline_group, bins=20, alpha=0.5, label='Offline (0 ratings)')
plt.xlabel('Average Rating')
plt.ylabel('Count')
plt.title('Distribution of Average Ratings: Online vs Offline Professors')
plt.legend()
plt.show()

#%%
# Visualization
plt.figure(figsize=(8, 5))
sns.boxplot(data=[online_group, offline_group], palette='pastel')
plt.xticks([0, 1], ['Online-heavy (≥2)', 'Offline-only (0)'])
plt.title('Average Ratings: Online-heavy vs. Offline Professors')
plt.ylabel('Average Rating')
plt.grid(True)
plt.tight_layout()
plt.show()


#%% Q5
#%%
# Drop missing values
df = num_data.dropna(subset=['Average Rating', 'Proportion who would take again'])

# Spearman correlation
corr, p_val = spearmanr(df['Average Rating'], df['Proportion who would take again'])

print("Spearman correlation: ", round(corr, 4))
print("p-value: ", p_val)

#%%
# Visualization
plt.figure(figsize=(8, 5))
sns.regplot(data=df, x='Average Rating', y='Proportion who would take again',
            scatter_kws={'alpha': 0.2}, line_kws={'color': 'red'})
plt.title('Take-Again Rate vs. Average Rating')
plt.xlabel('Average Rating')
plt.ylabel('Proportion Who Would Take Again')
plt.show()


#%% Q6
#%%
# Filter data
valid_pepper_data = num_data.dropna(subset=['Average Rating', 'Received a pepper?'])
peppered = valid_pepper_data[valid_pepper_data['Received a pepper?'] == 1]['Average Rating']
not_peppered = valid_pepper_data[valid_pepper_data['Received a pepper?'] == 0]['Average Rating']

def ourTestStatistic(x, y):
    return np.mean(x) - np.mean(y)
dataToUse_Q6 = (peppered, not_peppered)
pTest_Q6 = permutation_test(dataToUse_Q6, ourTestStatistic, n_resamples=10000, alternative='two-sided', random_state=19487111)

print('Test statistic:', pTest_Q6.statistic)
print('Exact p-value:', pTest_Q6.pvalue)

peppered_mean = peppered.mean()
peppered_median = peppered.median()

not_peppered_mean = not_peppered.mean()
not_peppered_median = not_peppered.median()

if peppered_mean > not_peppered_mean:
    print("\nPeppered Professors has higher mean rating.")
else:
    print("Not peppered Professors has higher mean rating.")
    
if peppered_median > not_peppered_median:
    print("Peppered Professors has higher median rating.")
else:
    print("Not peppered Professors has higher median rating.")
    

print("\nPeppered Professors:")
print("Mean Rating: ", round(peppered_mean, 3))
print("Median Rating: ", round(peppered_median, 3))

print("\nNot Peppered Professors:")
print("Mean Rating: ", round(not_peppered_mean, 3))
print("Median Rating: ", round(not_peppered_median,3))
# P-value is 0.00, which is smaller than alpha=0.05, we reject the null hypothesis.
# In other words, there is a difference in ratings between hot professors and not hot professors.
# Through comparing mean and median of two groups, we find out professors who are “hot” 
# receive higher ratings than those who are not.

#%%
# Plot the two distribution to see if they have similar distribution pattern
plt.figure(figsize=(8, 5))
plt.hist(peppered, bins=20, alpha=0.5, label='Peppered (Hot)')
plt.hist(not_peppered, bins=20, alpha=0.5, label='Unpeppered')
plt.xlabel('Average Rating')
plt.ylabel('Count')
plt.title('Distribution of Average Ratings: Peppered vs Unpeppered Professors')
plt.legend()
plt.tight_layout()
plt.show()

#%%
#Visualization
valid_pepper_data['Peppered'] = valid_pepper_data['Received a pepper?'].map({0: 'Not Peppered', 1: 'Peppered'})
plt.figure(figsize=(8, 5))
sns.boxplot(data=valid_pepper_data, x='Peppered', y='Average Rating', palette='pastel')
plt.title('Average Ratings: Peppered vs. Not Peppered Professors')
plt.ylabel('Average Rating')
plt.xlabel('Professor Status')
plt.grid(True)
plt.tight_layout()
plt.show()


#%% Q7
#%%
df = num_data.dropna(subset=['Average Rating', 'Average Difficulty'])

# Feature and target
X_7 = df[['Average Difficulty']]
y_7 = df['Average Rating']

np.random.seed(19487111)  
X_train, X_test, y_train, y_test = train_test_split(X_7, y_7, test_size=0.2, random_state=19487111)

# Fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
coef = model.coef_[0]
intercept = model.intercept_

# Print results
print(f"Regression Equation: Rating = {intercept:.4f} + ({coef:.4f}) × Difficulty")
print("R² (Test Set): ", round(r2, 4))
print("RMSE (Test Set): ", round(rmse, 4))

#%%
# Visualization
plt.figure(figsize=(6, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 45-degree line
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('Predicted vs Actual Average Ratings')
plt.grid(True)
plt.tight_layout()
plt.show()


#%% Q8
#%%
# Drop missing values
df = num_data.dropna(subset=[
    'Average Rating',
    'Average Difficulty',
    'Number of ratings',
    'Received a pepper?',
    'Proportion who would take again',
    'Number of online ratings',
    'Male gender',
    'Female'
])

# Define features and target
features = [
    'Average Difficulty',
    'Number of ratings',
    'Received a pepper?',
    'Proportion who would take again',
    'Number of online ratings',
    'Male gender',
    'Female'
]
X_8 = df[features]
y_8 = df['Average Rating']

# Train-test split
np.random.seed(19487111)  # replace with your N-number digits
X_train, X_test, y_train, y_test = train_test_split(X_8, y_8, test_size=0.2, random_state=19487111)

# Fit LassoCV
lasso = LassoCV(cv=5, random_state=19487111)
lasso.fit(X_train, y_train)

# Predict
y_pred = lasso.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Output
print("Lasso R²: ", round(r2, 4))
print("Lasso RMSE: ", round(rmse, 4))
print("Best alpha: ", lasso.alpha_)

# Coefficients
lasso_coef = pd.DataFrame({
    'Feature': features,
    'Coefficient': lasso.coef_
})
print("\nLasso Coefficients:")
print(lasso_coef)

#%%
#Visualization
plt.figure(figsize=(6, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 45-degree line
plt.xlabel("Actual Average Rating")
plt.ylabel("Predicted Average Rating")
plt.title("Predicted vs Actual Ratings (Lasso Regression)")
plt.show()


#%% Q9
#%%
# Drop missing values
df = num_data.dropna(subset=['Average Rating', 'Received a pepper?'])

# Define X and y
X_9 = df[['Average Rating']]
y_9 = df['Received a pepper?']

# Check class balance
print("Pepper counts:\n", y_9.value_counts(normalize=True))

# Train-test split
np.random.seed(19487111)  
X_train, X_test, y_train, y_test = train_test_split(X_9, y_9, test_size=0.2, random_state=19487111, stratify=y_9)

# Fit logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probabilities
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Evaluation
auc_9 = roc_auc_score(y_test, y_proba)
print("ROC AUC: ", round(auc_9, 4))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy_9 = (tp + tn) / (tp + tn + fp + fn)
sensitivity_9 = tp / (tp + fn)  # Recall
specificity_9 = tn / (tn + fp)
npv_9 = tn / (tn + fn)
precision_9 = tp / (tp + fp)

print("Q9 – Logistic Regression")
print("Accuracy:", round(accuracy_9, 4))
print("Sensitivity (Recall):", round(sensitivity_9, 4))
print("Specificity:", round(specificity_9, 4))
print("NPV:", round(npv_9, 4))
print("Precision:", round(precision_9, 4))

#%%
# Visualization
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Peppered', 'Peppered'], yticklabels=['Not Peppered', 'Peppered'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc_9:.4f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Predicting Pepper from Average Rating")
plt.legend()
plt.show()


#%% Q10
#%%
# Drop missing data
df = num_data.dropna(subset=[
    'Average Rating',
    'Average Difficulty',
    'Number of ratings',
    'Received a pepper?',
    'Proportion who would take again',
    'Number of online ratings',
    'Male gender',
    'Female'
])

# Define features and target
features = [
    'Average Rating',
    'Average Difficulty',
    'Number of ratings',
    'Proportion who would take again',
    'Number of online ratings',
    'Male gender',
    'Female'
]
X_10 = df[features]
y_10 = df['Received a pepper?']

# Train-test split (stratified for class balance)
np.random.seed(19487111)
X_train, X_test, y_train, y_test = train_test_split(X_10, y_10, test_size=0.2, random_state=19487111, stratify=y_10)

# Logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Evaluation
auc_10 = roc_auc_score(y_test, y_proba)
print("ROC AUC: ", round(auc_10, 4))

tn_pca, fp_pca, fn_pca, tp_pca = confusion_matrix(y_test, y_pred).ravel()
accuracy_10 = (tp_pca + tn_pca) / (tp_pca + tn_pca + fp_pca + fn_pca)
sensitivity_10 = tp_pca / (tp_pca + fn_pca)
specificity_10 = tn_pca / (tn_pca + fp_pca)
npv_10 = tn_pca / (tn_pca + fn_pca)
precision_10 = tp_pca / (tp_pca + fp_pca)

print("Q10 - Logistic Regression")
print("Accuracy:", round(accuracy_10, 4))
print("Sensitivity (Recall):", round(sensitivity_10, 4))
print("Specificity:", round(specificity_10, 4))
print("NPV:", round(npv_10, 4))
print("Precision:", round(precision_10, 4))


#%%
# Visualization
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Peppered', 'Peppered'], yticklabels=['Not Peppered', 'Peppered'])
plt.title('Confusion Matrix: Full Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc_10:.4f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Predicting Pepper from All Features')
plt.legend()
plt.show()


#%%
#Extra credit: Do professors in certain states get higher average ratings?
qual_data = pd.read_csv("rmpCapstoneQual.csv")
qual_data.columns = [
    'Major',
    'University',
    'State'
]

df = pd.concat([num_data, qual_data], axis=1)
df.columns = [
    'Average Rating', 'Average Difficulty', 'Number of ratings', 'Received a pepper?',
    'Proportion who would take again', 'Number of online ratings', 'Male gender', 'Female',
    'Major/Field', 'University', 'State'
]

# Drop missing values
df = df.dropna(subset=['Average Rating', 'State'])

# Group by state and compute mean rating and count
state_avg = df.groupby('State')['Average Rating'].agg(['mean', 'count']).reset_index()
state_avg = state_avg[state_avg['count'] >= 30]  # Only show states with >= 30 professors for stability

# Sort and plot
plt.figure(figsize=(12, 8))
sns.barplot(data=state_avg.sort_values('mean', ascending=False), x='mean', y='State', palette='viridis')
plt.title('Average Professor Ratings by State (States with ≥ 30 Professors)')
plt.xlabel('Average Rating')
plt.ylabel('State')
plt.tight_layout()
plt.show()








