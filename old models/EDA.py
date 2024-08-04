#!/usr/bin/env python
# coding: utf-8

# # Hepatitis B Classification

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


# # SMOTE

# In[38]:


X = ds.drop("class", axis=1)
y = ds["class"]


# In[ ]:





# In[39]:


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# In[40]:


#### Splitting the data int o training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


### Instantiating SMOTE
sm = SMOTE(random_state=42)

### Fit SMOTE to training data
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)


# In[41]:


#### Printing class disribution of original and resampled data
print("Class Distribution before Resampleing: ", y_train.value_counts())

print("\nClass Distribution afer Resampling", y_train_res.value_counts())


# In[42]:


# from sklearn.utils import resample

# X_resampled, y_resampled = resample(X_train_res, y_train_res, n_samples=100000, random_state=42)


# In[43]:


#y_resampled.value_counts()


# In[44]:


# X_resampled


# ## Correlation

# In[45]:


#### Corelation analysis
corr_matrix = ds.corr()
corr_matrix


# **Assuming we have the correlation matrix "corr_matrix" and wnt to change the figure soze**

# In[46]:


###Create the heatmap with Seaborn's sns.heatmap()

### Set the figure sixe using plt.figure()
plt.figure(figsize=(50, 20))

sns.heatmap(corr_matrix, annot=True, annot_kws={"fontsize": 18}) ### Adjust the fontsize as desired
plt.show()


# ## Scaling

# In[47]:


#### Data transformation
from sklearn.preprocessing import StandardScaler


# In[48]:


scaler = StandardScaler()


# In[49]:


num_cols = num_cols


# #### Create an empty DataFrame

# In[50]:


ds_scaled = pd.DataFrame()

for col in num_cols:
    ds_scaled[col] = ds[col]


# In[51]:


for col in num_cols:
    ds_scaled[col] = scaler.fit_transform(ds[col].values.reshape(-1, 1))


# In[52]:


ds.head()


# In[53]:


ds_scaled.head()


# ## Feature Importance

# In[54]:


from sklearn.ensemble import RandomForestClassifier


# In[55]:


X = ds.drop("class", axis=1)
y = ds["class"]


# #### Create a Random Forest Classifier

# In[56]:


rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

### Fit the classifier to the data
rf_classifier.fit(X, y)


# In[57]:


### Get feature importance from the trained model
feature_importances = rf_classifier.feature_importances_
feature_importances


# #### Create a DataFrame to visualize feature importances

# In[58]:


importance_ds = pd.DataFrame(
    {"Feature": X.columns, "Importance": feature_importances}
)
importance_ds= importance_ds.sort_values(by="Importance", ascending=False)

## Printing the DataFrame to see feature importances
importance_ds


# ## Principal Component Analysis (PCA)

# In[59]:


from sklearn.decomposition import PCA

X = ds.drop("class", axis=1)
n_components = 2
pca = PCA(n_components=n_components)


# ### Fit and trandform the data using PCA

# In[60]:


X_pca = pca.fit_transform(X)


# In[61]:


### Create a DataFrame with the transformed data
pca_ds = pd.DataFrame(
    data=X_pca, columns=[f"PC{i+1}" for i in range(n_components)]
)

pca_ds.head()


# ## Visualizations

# In[62]:


ds = pd.read_csv("hepatitis_dataset.csv")


# In[ ]:





# In[63]:


#### Get the list of column names in the DataFrame

column_names = ds.columns

# Loop through the columns
for column in column_names:
    
    # Check if the column is categorical (object or category dtype)
    if ds[column].dtype == "object" or pd.api.types.is_categorical_dtype(ds[column]):
        
        ## Count the occurences of each category in the column
        category_counts = ds[column].value_counts()
        
        ## Create a pie chrt for the column
        plt.figure(figsize=(4, 4)) # Adjust the figur esize as desired
        plt.pie(
            category_counts, labels=category_counts.index, autopct="%1.1f%%", startangle=90
        )
        
        plt.title(f"Pie Chart for {column}")
        plt.axis("equal") # Equal aspect ration ensures that pie is drawn as a circle
        plt.show()


# In[64]:


column_names


# In[65]:


# Loop through the columns
for column in column_names:
    # Check if the column is categorical (object or category dtype)
    if ds[column].dtype == 'object' or pd.api.types.is_categorical_dtype(ds[column]):
        # Create a bar plot for categorical columns
        plt.figure(figsize=(6, 4))  # Adjust the figure size as desired
        sns.countplot(data=ds, x=column)
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.title(f'Bar Plot for {column}')
        plt.xticks(rotation=45)  # Rotate x-axis labels if needed
        plt.show()      
        
    elif ds[column].dtype == ['int64', 'float64']:
        # Create a histogram for numerical columns
        plt.figure(figsize=(6, 4))  # Adjust the figure size as desired
        plt.hist(df[column], bins=10)  # You can adjust the number of bins
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.title(f'Histogram for {column}')
        plt.show()


# In[70]:


for column in column_names:
    if pd.api.types.is_numeric_dtype(ds[column]):
        plt.figure(figsize=(6, 4))  # Adjust the figure size as desired
        plt.scatter(ds[column], ds['class'])
        plt.xlabel(column)
        plt.ylabel('Class')
        plt.title(f'Scatter Plot for {column} vs. Class')
        plt.grid()
        plt.show()


# In[ ]:




