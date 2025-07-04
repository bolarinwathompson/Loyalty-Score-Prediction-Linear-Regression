# Linear Regression- ABC Groceries Test


# Import Required Python Packages

import pandas as pd
import pickle 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV


# Import Sample Data

data_for_model = pickle.load(open("regression_modelling.p", "rb"))


# Drop Unnecessary Coulumns

data_for_model.drop("customer_id", axis =1, inplace = True)







