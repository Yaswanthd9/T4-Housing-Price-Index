#%%

from mysqlx import Column
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm 
from statsmodels.formula.api import ols
from statsmodels.formula.api import glm
from statsmodels.formula.api import logit
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
fatcalfit = LogisticRegression(max_iter= 1000)
from sklearn.model_selection import train_test_split
#%%
## Import finalized dataset as .csv##
pricedf1= pd.read_csv('FinalDC.csv')
# %%
pricedf1.head()
# %%
print(pricedf1)
#%%
pricedf1['PRICE'].describe()
#%%
pricedf1['metro25']= pricedf1['.25metro']
pricedf1['metro50']= pricedf1['.50metro']
#%%
formula1=('PRICE ~ metro25  + BEDRM + ROOMS + BATHRM + AC ')
#%%
modelTitanicAllLogitFit= ols(formula=formula1, data=pricedf1).fit()
modelTitanicAllLogitFit.summary()
print( type(modelTitanicAllLogitFit) )
print( modelTitanicAllLogitFit.summary() )
# %%

+ AC + NUM_UNITS + LANDAREA
