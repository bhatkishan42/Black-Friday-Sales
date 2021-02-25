# Black-Friday-Sales-Prediction


# Black Friday

 - Data is taken from Kaggle and from other competitions

 - A retail company “ABC Private Limited” wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month.
 - The data set also contains customer demographics (age, gender, marital status, city_type, stay_in_current_city), product details (product_id and product category) and Total purchase_amount from last month.
 - Now, they want to build a model to predict the purchase amount of customer against various products which will help them to create personalized offer for customers against different products.


 - Let's select are pre-requsits library and load them

```python
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

Data
- Variable                  Definition
- User_ID	                User ID
- Product_ID                Product ID
- Gender	                Sex of User
- Age	                    Age in bins
- Occupation	            Occupation (Masked)
- City_Category	            Category of the City (A,B,C)
- Stay_In_Current_City_Years	Number of years stay in current city
- Marital_Status	        Marital Status
- Product_Category_1	    Product Category (Masked)
- Product_Category_2	    Product may belongs to other category also (Masked)
- Product_Category_3	    Product may belongs to other category also (Masked)
- Purchase	Purchase Amount (Target Variable)
 * *Your model performance will be evaluated on the basis of your prediction of the purchase amount for the test data -(test.csv), which contains similar data-points as train except for their purchase amount. Your submission needs to be in the format as shown in "SampleSubmission.csv".*

We at our end, have the actual purchase amount for the test dataset, against which your predictions will be evaluated. Submissions are scored on the root mean squared error (RMSE). RMSE is very common and is a suitable general-purpose error metric. Compared to the Mean Absolute Error, RMSE punishes large errors:


 Where y hat is the predicted value and y is the original value.


#### Data Overview
 *Dataset has 537577 rows (transactions) and 12 columns (features) as described below:*

 - User_ID: Unique ID of the user. There are a total of 5891 users in the dataset.
 - Product_ID: Unique ID of the product. There are a total of 3623 products in the dataset.
 - Gender: indicates the gender of the person making the transaction.
 - Age: indicates the age group of the person making the transaction.
 - Occupation: shows the occupation of the user, already labeled with numbers 0 to 20.
 - City_Category: User's living city category. Cities are categorized into 3 different categories 'A', 'B' and 'C'.
 - Stay_In_Current_City_Years: Indicates how long the users has lived in this city.
 - Marital_Status: is 0 if the user is not married and 1 otherwise.
 - Product_Category_1 to _3: Category of the product. All 3 are already labaled with numbers.
 - Purchase: Purchase amount.


# ML requiremnets
Shap
Eli5 (Permutation)
OneHotEncode (assignning values)
Clustermap
Linear Regression

### Reading data


- Importing  data and reading

# ```python
data_train_path='/kaggle/input/black-friday/train.csv'
data_test_path='/kaggle/input/black-friday/test.csv'
df_a = pd.read_csv(data_train_path)
df_b = pd.read_csv(data_test_path)
```
We'll work and explore the above data sets.

```python
df1 = df_a
df11 = df_b
```
```python
list(df1.columns)
```
```python
df1.head(5)
```
```python
df_b_a = df_a.copy()
df_b_b = df_b.copy()
```

```python
list(df1.columns)
````
```python
df1.info()
```

 Lets find the unique variable 


  - As we can see that User_ID ,Product_ID  has large no of unique values due to which it is useless for machine learning 
   in later stage we'll drop it 

```python
df1.nunique()

 -Product_Category_2             72344
 -Product_Category_3            162562
 large no of data is missing we need to find out the pattern and decide to weather to drop or impute or fill  
```
```python
df1.isna().sum()
```
```pythonlist(df1.select_dtypes(include = np.number))

# %% [markdown]
# - Dropping Product_Category_3 as huge no off data is missing
# - And we are imputing Product_Category_2 
# - Method == ***Median()***

```pythondf1['Product_Category_2'].fillna(df1.Product_Category_2.median(), inplace=True)
df1.isna().sum()
# drop Product_Category_3 
df1.drop('Product_Category_3',axis=1,inplace = True)
```
```python
df1.Stay_In_Current_City_Years
df1.Stay_In_Current_City_Years.replace("4+","4")
```

 4.1 as to indicate +4 yrs of stay 

```python
df1.Stay_In_Current_City_Years = df1.Stay_In_Current_City_Years.replace("4+","4.1").astype("float")
print(df1.Stay_In_Current_City_Years)
```
 Using mapping we place male as 1 and female as  2 for purpose to use this efficently in data analysis

```python
df1['GEN_1'] = df1.Gender.map({'M':1,'F':2})
```

```python
list(df1.columns)
```

## Exploratory Data Analysis


lets get on with explorattion and look out for any key patterns

```python
# sns.pairplot(df1.drop(columns = ['User_ID','Product_ID']).select_dtypes(include = np.number))
```
# %% [markdown]
# - Its checks out more no of male are present during the black friday sale copared to females

# %% [code]
plt.xlabel('Plot Number')
plt.title('Gender count')
sns.countplot(x = df1.GEN_1,data = df1)

# %% [markdown]
# So from the below plot & list we can say that on an average ppl staying for around 4+ yrs are the one to buy most from sale,followed by 3yrs,2yrs,1 yrs respectively

# %% [code]
df1.groupby('Stay_In_Current_City_Years').Stay_In_Current_City_Years.sum().sort_values(ascending =False)

# %% [code]
df1.groupby('Stay_In_Current_City_Years').Stay_In_Current_City_Years.sum().sort_values(ascending = False).plot(kind = 'bar').set_xlabel("NO. of years")

# %% [markdown]
# #### Inference
# - Looking at below plots we can see that age grp of **26-35** are the highest in the grp following **36-45**,**18-25**,**51-55** respectively
# - In city category we can see that category **B** is more followed by **A** and **C** respectively
# - Regarding maritial status Unmarried ppls do buy more compared to Other 

# %% [code]
sns.countplot(df1.Age)

# %% [code]
sns.countplot(df1.City_Category)

# %% [code]
sns.countplot(df1.Marital_Status)

# %% [markdown]
# 
# P00265242,P00112142,P00025442,P00110742,P00046742  
# - these are top 5 selling products during black friday sale
# 

# %% [code]
df1.Product_ID.value_counts().head(50)

# %% [code]
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.xticks(rotation = 90)
df1.Product_ID.value_counts().head(50).plot(kind = 'bar')

# %% [markdown]
# - Product_Category_2 has the product 9.0 selling highest followed by product 8

# %% [code]
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.xticks(rotation = 90)
df1.Product_Category_2.value_counts(ascending = False).plot(kind = 'bar')

# %% [code]
df1.groupby('Product_Category_1')['Product_ID'].nunique().plot(kind = 'bar')

# %% [code]
sns.heatmap(df1.drop(columns=['User_ID','Product_ID']).select_dtypes(include = np.number).corr(method = "kendall"),cmap='Greys',annot = True)

# %% [code]
sns.clustermap(df1.drop(columns=['User_ID','Product_ID']).select_dtypes(include = np.number).corr(method = "kendall"),cmap='Greys' )

# %% [code]
 data_1 = df1.iloc[:,-4:-1]
print(pd.DataFrame(data_1))

# %% [code]
df11.isna().sum()

# %% [markdown]
# ## Statistical analysis

# %% [markdown]
# ##### OneHotEncoder

# %% [markdown]
# - We can also us pd_get_dummies to get encoding but we will follow traditional approach

# %% [code]
print(df11.shape)

print(df1.shape)

# %% [code]
df11['Product_Category_2'].fillna(df11.Product_Category_2.median(), inplace=True)
df11.isna().sum()
# drop Product_Category_3 
df11.drop('Product_Category_3',axis=1,inplace = True)

# %% [markdown]
# ###### We Are using test data set for analysis 

# %% [code]
# drop User_ID and Product_ID
df1.drop(columns = ['User_ID','Product_ID'],inplace = True)

# %% [code]
df1.Purchase 

# %% [code]
# one-hot encoding of categorical features
from sklearn.preprocessing import OneHotEncoder
# get categorical features and review number of unique values
cat=df1.select_dtypes(exclude= np.number)
print("Number of unique values per categorical feature:\n", cat.nunique())
# use one hot encoder
enc = OneHotEncoder(sparse=False).fit(cat)
cat_encoded = pd.DataFrame(enc.transform(cat))
cat_encoded.columns = enc.get_feature_names(cat.columns)
# merge with numeric data
num = df1.drop(columns =cat.columns)
df2 = pd.concat([cat_encoded, num], axis=1)
df2.head()

# %% [code]
df2.columns

# %% [markdown]
# findings : Rmse is  smaller than median value indicating good model

# %% [code]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# isolate X and y variables, and perform train-test split
X = df2.drop(columns='Purchase')
y = df2['Purchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# predictions
preds = model.predict(X_test)

# evaluate model using RMSE
print("Linear regression model RMSE: ", np.sqrt(mean_squared_error(y_test, preds)))
print("Median value of target variable: ", y.median())

# %% [code]
- age,gender,city category,product category , occupation , stay , martital are listed according to the weights

# %% [markdown]
# - 

# %% [code]
import eli5 
from eli5.sklearn import  PermutationImportance
perm =PermutationImportance(model,random_state=1).fit(X_train,y_train)
eli5.show_weights(perm,feature_names = X_test.columns.tolist(),top = 20)

# %% [code]
import shap
# calculate shap values 
ex = shap.Explainer(model, X_train)
shap_values = ex(X_test)
# plot
plt.title('SHAP summary for NumStorePurchases', size=16)
shap.plots.beeswarm(shap_values, max_display=5)
