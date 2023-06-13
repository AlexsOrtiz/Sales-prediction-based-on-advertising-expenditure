import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

#Load the data from a CSV file

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv")
#Print the first few rows of the data

print(data.head())
#Check for missing values in the data

print(data.isnull().sum())
#Create a scatter plot of Sales vs. TV with a trendline

figure = px.scatter(data_frame=data, x="Sales", y="TV", size="TV", trendline="ols")
figure.show()
#Create a scatter plot of Sales vs. Newspaper with a trendline

figure = px.scatter(data_frame=data, x="Sales", y="Newspaper", size="Newspaper", trendline="ols")
figure.show()
#Create a scatter plot of Sales vs. Radio with a trendline

figure = px.scatter(data_frame=data, x="Sales", y="Radio", size="Radio", trendline="ols")
figure.show()
#Calculate the correlation between variables and Sales

correlation = data.corr()
print(correlation["Sales"].sort_values(ascending=False))
#Select the features and target variable

X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]
#Normalize the numerical features

scaler = StandardScaler()
X = scaler.fit_transform(X)
#Split the data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Create a Ridge regression model

model = Ridge()
#Perform cross-validation

cross_val_score(model, X_train, y_train, cv=5)
#Perform grid search to find the best hyperparameters

param_grid = {
'alpha': [0.1, 1.0, 10.0],
}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
#Evaluate the model's performance using mean squared error

mse = mean_squared_error(y_test, grid_search.predict(X_test))
print("Mean Squared Error:", mse)
#Make a prediction

features = scaler.transform(np.array([[230.1, 37.8, 69.2]]))
print(grid_search.predict(features))
