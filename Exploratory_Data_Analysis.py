#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis for Hepatitis B Classification

# #### Importing project dependencies

# In[1]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import re


# #### Data Pre-Processing and Exploratory Data Analyis (EDA)

# In[2]:


ds = pd.read_csv("hepatitis_dataset.csv")


# In[3]:


ds.head()


# #### Set the 'max_rows' parameter to a large number to display all rows

# In[4]:


pd.set_option("display.max_rows", None)


# In[5]:


ds.head()


# #### Set the 'max_columns' parameter to a large number to display all rows
# 

# In[6]:


pd.set_option("display.max_columns", None)


# In[7]:


ds.head()


# #### Displaying details of dataset

# In[8]:


ds.info()


# In[9]:


ds.dtypes


# ## Label Encoding

# #### Separating categorical variables from numeric variables

# In[10]:


cat_cols = ds.select_dtypes(include=["object", "bool"]).columns.tolist()


# In[11]:


cat_cols


# In[12]:


from sklearn.preprocessing import LabelEncoder


# In[13]:


encoder = LabelEncoder()

for i in cat_cols:
    encoder.fit(ds[i])
    ds[i] = encoder.transform(ds[i])


# In[14]:


ds.head()


# In[15]:


ds.info()


# ## Null Value Analysis

# In[16]:


ds.isna().any()


# In[17]:


ds.isna().sum()


# ### Filling the null values

# In[18]:


ds.fillna(ds.mode(), inplace=True)
ds.fillna(ds.mean(), inplace=True)


# In[19]:


ds.isna().any()


# ## Unique Values

# In[20]:


ds.nunique()


# In[ ]:





# ## Outliers

# In[21]:


def find_outliers_iqr(data):
    # Sort the data in ascending order
    sorted_data = sorted(data)
    
    # Calculate the 25th and 75th percentiles (Q1 and Q3)
    q1 = sorted_data[int(len(sorted_data) * 0.25)]
    q3 = sorted_data[int(len(sorted_data) * 0.75)]
    
    # Calculate the interquartile range (IQR)
    iqr = q3 - q1
    
    # Calculate the lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Find outliers
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    
    return outliers


# In[22]:


num_cols = ds.select_dtypes(include=["int64", "float64"]).columns.tolist()


# In[23]:


num_cols


# In[24]:


for col in num_cols:
    print(col)
    print("-"*70)
    print(find_outliers_iqr(ds[col]))


# ## Feature Selection

# In[25]:


from sklearn.feature_selection import SelectKBest, f_regression


# #### Create a SelectKBest Object

# In[26]:


selector = SelectKBest(score_func=f_regression, k=5)


# #### Select the features and target variable

# In[27]:


X = ds.iloc[:, :-1]
y = ds.iloc[:, -1]


# #### Fit the select to te data

# In[28]:


selector.fit(X, y)


# In[29]:


## Get the selected features
selector.get_support()


# In[30]:


selector.scores_


# In[31]:


### Get the selected features

selected_feature_indices = selector.get_support(
    indices=True
)
selected_features = ds.columns[selected_feature_indices]

### Print only the columns with True values
selected_ds = ds[selected_features]


# In[32]:


selected_ds.head()


# In[33]:


selected_feature_indices


# In[34]:


selected_features


# ## Class Distributions

# #### count the number of instances in each class
# 

# In[35]:


class_counts = ds["class"].value_counts()


# In[36]:


#### Print the class disribution
print("Class Distribution: ", class_counts)


# **Assuming we already have the DataFrame "ds" and "class_counts" as computed in the code**

# #### Plot the pie chart

# In[37]:


plt.figure(figsize=(5, 5))
plt.pie(class_counts, labels=["Lived", "Died"], autopct="%1.1f%%", colors=["skyblue", "lightcoral"])
plt.title("Class Distribution")
plt.axis("equal") ## Equl aspect ration ensures the pie chart is circular
plt.show()


# ## Correlation

# In[38]:


#### Corelation analysis
corr_matrix = ds.corr()
corr_matrix


# **Assuming we have the correlation matrix "corr_matrix" and wnt to change the figure soze**

# ### Create the heatmap with Seaborn's sns.heatmap()
# 

# In[39]:


### Set the figure sixe using plt.figure()
plt.figure(figsize=(50, 20))

sns.heatmap(corr_matrix, annot=True, annot_kws={"fontsize": 18}) ### Adjust the fontsize as desired
plt.show()


# ## Feature Importance

# In[40]:


from sklearn.ensemble import RandomForestClassifier


# In[41]:


X = ds.drop("class", axis=1)
y = ds["class"]


# #### Create a Random Forest Classifier

# In[42]:


rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

### Fit the classifier to the data
rf_classifier.fit(X, y)


# ### Get feature importance from the trained model
# 

# In[43]:


feature_importances = rf_classifier.feature_importances_
feature_importances


# #### Create a DataFrame to visualize feature importances

# In[44]:


importance_ds = pd.DataFrame(
    {"Feature": X.columns, "Importance": feature_importances}
)
importance_ds= importance_ds.sort_values(by="Importance", ascending=False)

## Printing the DataFrame to see feature importances
importance_ds

