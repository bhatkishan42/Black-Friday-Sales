
# Data analysis report,codes are available in python notebook named as (black-friday-project.ipynb) in this repository

# Black-Friday-Sales-Prediction


 - Data is taken from Kaggle and from other competitions

 - A retail company “ABC Private Limited” wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month.
 - The data set also contains customer demographics (age, gender, marital status, city_type, stay_in_current_city), product details (product_id and product category) and Total purchase_amount from last month.
 - Now, they want to build a model to predict the purchase amount of customer against various products which will help them to create personalized offer for customers against different products.



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



