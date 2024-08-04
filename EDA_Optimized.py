#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis for Hepatitis B Classification
# 

# ### Importing project dependencies
# 

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestClassifier


# ### Set display options for better visualization

# In[2]:


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


# ### Data Loading
# 

# In[3]:


# Load the dataset
ds = pd.read_csv("hepatitis_dataset.csv")


# In[4]:


# Display the first few rows
ds.head(20)


# ### Displaying details of dataset

# In[5]:


# Information about the dataset
ds.info()


# ### Check for missing values

# In[6]:


ds.isna().sum()


# ### Label Encoding for categorical variables

# In[7]:


# Identify categorical columns
cat_cols = ds.select_dtypes(include=["object", "bool"]).columns.tolist()
print("Categorical Columns:", cat_cols)


# In[8]:


# Initialize LabelEncoder
encoder = LabelEncoder()


# In[9]:


# Encode categorical columns
for col in cat_cols:
    ds[col] = encoder.fit_transform(ds[col])


# In[10]:


# Verify encoding
ds.head()


# ### Handling Missing Values

# In[11]:


# Fill missing values
ds.fillna(ds.mode().iloc[0], inplace=True)  # For categorical data
ds.fillna(ds.mean(), inplace=True)  # For numerical data


# In[12]:


# Verify that there are no missing values
print("Missing values after filling:", ds.isna().sum().sum())


# In[13]:


#### Unique Values
print("Number of unique values per column:")
print(ds.nunique())


# ### Outliers Detection

# In[14]:


def find_outliers_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data < lower_bound) | (data > upper_bound)]


# ### Identify numeric columns

# In[15]:


num_cols = ds.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Print outliers for each numeric column
for col in num_cols:
    outliers = find_outliers_iqr(ds[col])
    print(f"Outliers in {col}: {len(outliers)} instances")


# ### Feature Selection

# In[16]:


# SelectKBest to select top features
X = ds.drop(columns=["class"])
y = ds["class"]
selector = SelectKBest(score_func=f_regression, k=5)
selector.fit(X, y)


# In[17]:


# Get the selected features
selected_features = X.columns[selector.get_support()]
print("Selected Features:", selected_features)


# ### Class Distributions
# 

# In[18]:


class_counts = ds["class"].value_counts()
print("Class Distribution:\n", class_counts)


# ### Plot the class distribution

# In[19]:


plt.figure(figsize=(5, 5))
plt.pie(class_counts, labels=["Lived", "Died"], autopct="%1.1f%%", colors=["skyblue", "lightcoral"])
plt.title("Class Distribution")
plt.axis("equal")  # Equal aspect ratio ensures the pie chart is circular
plt.show()


# #### Correlation Matrix

# In[20]:


# Calculate the correlation matrix
corr_matrix = ds.corr()


# In[21]:


# Plot the heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f')
plt.show()



# ### Feature Importance with Random Forest
# 

# In[22]:


rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X, y)


# ### Get feature importances
# 
# 

# In[23]:


feature_importances = rf_classifier.feature_importances_
importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)


# In[24]:


importance_df


# In[ ]:




