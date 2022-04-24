#%%
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Read King County, WA Housing Data csv
url = "https://raw.githubusercontent.com/garrettwilliams90/Housing_Analysis/main/data/kc_house_data.csv"
df = pd.read_csv(url)
df.head()
# %%
y = df['price']
xx = df.drop('price', axis = 1)
xx = sm.add_constant(xx)
df = pd.get_dummies(df, columns=['bedrooms'], drop_first=True)
df = pd.get_dummies(df, columns=['grade'], drop_first=True)
df = pd.get_dummies(df, columns=['view'], drop_first=True)
df = pd.get_dummies(df, columns=['waterfront'], drop_first=True)
print(df.head())
print('Done, continue.')
#%%
#Create an empty dictionary that will be used to store our results
function_dict = {'predictor': [], 'r-squared':[]}
#Iterate through every column in X
for col in X.columns:
    #Create a dataframe called selected_X with only the 1 column
    selected_X = Xx[[col]]
    #Fit a model for our target and our selected column 
    model = sm.OLS(y, sm.add_constant(selected_X)).fit()
    #Predict what our target would be for our model
    y_preds = model.predict(sm.add_constant(selected_X))
    #Add the column name to our dictionary
    function_dict['predictor'].append(col)
    #Calculate the r-squared value between the target and predicted target
    r2 = np.corrcoef(y, y_preds)[0, 1]**2
    #Add the r-squared value to our dictionary
    function_dict['r-squared'].append(r2)
    
#Once it's iterated through every column, turn our dictionary into a DataFrame and sort it
function_df = pd.DataFrame(function_dict).sort_values(by=['r-squared'], ascending = False)
#Display only the top 5 predictors
print(function_df.head())
# %%
