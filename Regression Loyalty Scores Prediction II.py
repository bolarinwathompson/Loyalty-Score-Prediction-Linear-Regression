# Linear Regression- Basic Templates


# Import Required Python Packages

from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd


# Import Sample Data
my_df = pd.read_csv("sample_data_regression.csv")


# Split the Data into input and Output Data
X = my_df.drop(["output"], axis = 1)
y = my_df["output"]

# Split the Data into Test and Train Datasets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Instantiate our Model Object

regression = LinearRegression()

# Train our Model

regression.fit(X_train, y_train)

# Assess Model Accuracy

y_pred = regression.predict(X_test)
r2_score(y_test, y_pred,)

print(r2_score)









