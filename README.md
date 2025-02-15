# CODEALPHA-TASK-3
### CAR PRICE PRIDICTION WITH MACHINE LEARNING ###
This code, executed in a Jupyter Notebook, is used to predict the selling price of a car using a dataset. It performs data preprocessing, trains a random forest regression model, and evaluates its performance using various metrics. Here’s a detailed explanation of the code:

 Step 1: **Importing Required Libraries**
The necessary libraries for data manipulation, machine learning, and visualization are imported:
- **pandas** is used for reading and handling data in tabular form.
- **numpy** is used for numerical operations and array manipulation.
- **sklearn.model_selection** provides functions to split the data into training and testing sets.
- **sklearn.preprocessing** includes tools like LabelEncoder, which converts categorical data into numeric values.
- **sklearn.ensemble** contains the RandomForestRegressor, a machine learning model that is used for regression tasks.
- **sklearn.metrics** includes functions to evaluate the model’s performance, such as Mean Absolute Error, Mean Squared Error, and R² Score.
- **matplotlib** and **seaborn** are used for creating visualizations like graphs and plots.

 Step 2: **Reading the Dataset**
The dataset is loaded from a CSV file using the pandas `read_csv()` function. After the dataset is loaded, the first few rows of the dataset are displayed using the `head()` function to provide a glimpse of the data.

 Step 3: **Handling Missing Values**
The code checks for missing values using `isnull().sum()` and handles them accordingly:
- For numerical columns (like Present_Price, Driven_kms, and Year), any missing values are replaced with the mean of the respective column using the `fillna()` function.
- For categorical columns (like Fuel_Type, Selling_type, Transmission, and Owner), missing values are replaced with the most frequent value (mode) of that column.

 Step 4: **Encoding Categorical Data**
Since machine learning models require numerical data, the code uses **LabelEncoder** to convert categorical variables into numeric labels:
- **Fuel_Type**, **Selling_type**, **Transmission**, and **Owner** are the categorical columns that are transformed into numeric values. This step ensures that the machine learning model can work with these columns.

 Step 5: **Dropping Irrelevant Column**
The **Car_Name** column is dropped from the dataset. Since this column likely contains irrelevant information for predicting the selling price, it is not needed for the model.

 Step 6: **Splitting the Data into Features and Target**
The dataset is divided into the feature set (**X**) and the target variable (**y**):
- **X** includes all columns except for **Selling_Price**, which are the independent variables used to predict the selling price.
- **y** is the target variable, which is the **Selling_Price** column. This is the value the model aims to predict.

 Step 7: **Splitting Data into Training and Testing Sets**
The **train_test_split()** function splits the data into training and testing sets. Typically, a dataset is divided into 80% for training the model and 20% for testing the model’s performance. This allows the model to learn from the training data and then be evaluated on the unseen test data.

 Step 8: **Training the Random Forest Model**
A **RandomForestRegressor** model is created with 100 decision trees and is trained on the training data (**X_train**, **y_train**). The model learns the relationship between the features and the target variable during this step.

 Step 9: **Making Predictions**
After the model is trained, predictions are made using the **predict()** function on the test set (**X_test**). The predicted selling prices are stored in the variable **y_pred**.

 Step 10: **Evaluating the Model**
The model’s performance is evaluated using several metrics:
- **Mean Absolute Error (MAE)**: Measures the average of absolute differences between actual and predicted values. Lower values indicate better performance.
- **Mean Squared Error (MSE)**: Similar to MAE but penalizes larger errors more significantly.
- **Root Mean Squared Error (RMSE)**: The square root of MSE, which brings the error value back to the original scale of the data.
- **R² Score**: Measures how well the model explains the variance in the target variable. A score closer to 1 indicates a good fit.

 Step 11: **Visualizing the Results**
A regression plot is created using **seaborn** to compare the actual vs. predicted selling prices. The plot helps visualize how well the model’s predictions align with the actual data points.

In summary, this code processes a car sales dataset, fills missing data, encodes categorical variables, trains a random forest regression model, evaluates its performance, and visualizes the results in a Jupyter Notebook environment. The goal is to predict the selling price of cars based on various features like their present price, driven kilometers, year of manufacture, and others.
