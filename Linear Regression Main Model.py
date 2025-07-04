# Logistic Regression - ABC Grocery Task

# Import Required Packages
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV

# -------------------------------------------------------------------
# Import Sample Data
data_for_model = pickle.load(open("regression_modelling.p", "rb"))

# -------------------------------------------------------------------
# Drop Unnecessary Columns
data_for_model.drop("customer_id", axis=1, inplace=True)

# -------------------------------------------------------------------
# Shuffle the Data
data_for_model = shuffle(data_for_model, random_state=42)

# -------------------------------------------------------------------
# Deal with Missing Data
print("Missing values per column:")
print(data_for_model.isna().sum())
data_for_model.dropna(how="any", inplace=True)

# -------------------------------------------------------------------
# Deal with Outliers using a Boxplot Approach
outlier_columns = ["distance_from_store", "total_sales", "total_items"]
for column in outlier_columns:
    lower_quantile = data_for_model[column].quantile(0.25)
    upper_quantile = data_for_model[column].quantile(0.75)
    iqr = upper_quantile - lower_quantile
    iqr_extended = iqr * 2
    min_border = lower_quantile - iqr_extended 
    max_border = upper_quantile + iqr_extended 
    outliers = data_for_model[(data_for_model[column] < min_border) | (data_for_model[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")
    data_for_model.drop(outliers, inplace=True)
    print(f"After dropping outliers for {column}, data_for_model shape: {data_for_model.shape}")

# -------------------------------------------------------------------
# Split Input Variables and Output Variable
X = data_for_model.drop(["customer_loyalty_score"], axis=1)
y = data_for_model["customer_loyalty_score"]

# -------------------------------------------------------------------
# Split out Training and Test Sets (this step ensures X_test is defined)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------------------
# Feature Selection: One-Hot Encoding for Categorical Variable(s)
categorical_vars = ["gender"]

# Use sparse_output=False for scikit-learn 1.2+; change to sparse=False if using an earlier version.
one_hot_encoder = OneHotEncoder(sparse_output=False, drop="first")

# Fit the encoder on the training set's categorical column and transform both training and test sets.
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# Get the encoded feature names using the new method.
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# Convert the encoded arrays into DataFrames (keeping the original indices for proper alignment).
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoder_feature_names, index=X_train.index)
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoder_feature_names, index=X_test.index)

# Remove the original categorical column(s) from X_train and X_test.
X_train_clean = X_train.drop(columns=categorical_vars)
X_test_clean = X_test.drop(columns=categorical_vars)

# Concatenate the cleaned DataFrames with their corresponding encoded DataFrames.
X_train = pd.concat([X_train_clean, X_train_encoded_df], axis=1)
X_test = pd.concat([X_test_clean, X_test_encoded_df], axis=1)

# Feature Selection


regressor = LinearRegression()
feature_selector = RFECV(regressor)

# Fit the RFECV on the training set
fit = feature_selector.fit(X_train, y_train)
optimal_feature_count = feature_selector.n_features_

print(f"Optimal number of features: {optimal_feature_count}")

# Select features based on the feature_selector's support mask
X_train = X_train.loc[:, feature_selector.get_support()]
X_test = X_test.loc[:, feature_selector.get_support()]

# Retrieve the mean cross-validation scores from cv_results_
mean_scores = feature_selector.cv_results_["mean_test_score"]

# Plot the model scores against the number of features
plt.plot(range(1, len(mean_scores) + 1), mean_scores, marker="o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection Using RFE\nOptimal number of Features is {optimal_feature_count}")
plt.tight_layout()
plt.show()


# Model Training 

regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predict The Test Score
y_pred = regressor.predict(X_test)

# Calculate the R-Squared
r_squared = r2_score(y_test, y_pred)
print(r_squared)

# Cross Validation 

cv = KFold(n_splits = 4, shuffle = True, random_state = 42)

cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = "r2")
cv_scores.mean()


# Calculate the Adjusted R-Square

num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 -(1 - r_squared)*(num_data_points - 1)/ (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)

# Extract Model Coefficients

coefficients = pd.DataFrame(regressor.coef_)
input_variable_names = pd.DataFrame(X_train.columns)
summary_stat = pd.concat([input_variable_names, coefficients], axis = 1)
summary_stat.columns = ["input_variable", "coefficient"]


# Extract the Model Intercept

regressor.intercept_

print(regressor.intercept_)










