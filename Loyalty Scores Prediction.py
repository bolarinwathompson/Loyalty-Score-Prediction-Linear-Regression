# Creating Data for ABC Regression Task

# Import necessary Packages

import pandas as pd
import pickle

# Import the needed Data

loyalty_scores = pd.read_excel("grocery_database.xlsx", sheet_name= "loyalty_scores")
customer_details = pd.read_excel("grocery_database.xlsx", sheet_name= "customer_details")
transactions = pd.read_excel("grocery_database.xlsx", sheet_name= "transactions")

#Create the customer level database

data_for_regression = pd.merge(customer_details, loyalty_scores, how = "left", on = "customer_id")


sales_summary = transactions.groupby("customer_id").agg({
    "sales_cost": "sum",
    "num_items": "sum",
    "transaction_id": "count",
    "product_area_id": "nunique"
}).reset_index()


sales_summary.columns = ["customer_id", "total_sales", "total_items", "transaction_count", "product_area_count" ]

sales_summary["avregae_basket_value"] = sales_summary["total_sales"] / sales_summary["transaction_count"]


data_for_regression = pd.merge(
data_for_regression, sales_summary, how = "inner", on = "customer_id")

regression_modelling = data_for_regression.loc[data_for_regression["customer_loyalty_score"].notna()]
regression_scoring = data_for_regression.loc[data_for_regression["customer_loyalty_score"].isna()]
regression_scoring.drop(["customer_loyalty_score"], axis = 1, inplace = True)



pickle.dump(regression_modelling, open("DATA SCIENTIST/abc_regression_modelling.p", "wb"))
pickle.dump(regression_scoring, open("DATA SCIENTIST/abc_regression_scoring.p", "wb"))
