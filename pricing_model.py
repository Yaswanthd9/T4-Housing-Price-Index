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






#### OTHER SCATTER PLOTS ####
#Regression Plot
sns.regplot(x="BATHRM", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"}, x_jitter=10, line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.show()
# %%
sns.regplot(x="ROOMS", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"}, x_jitter=10, line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Rooms")
plt.ylabel("Price")
plt.show()
#%%
#%%
sns.scatterplot(x="distance", y="PRICE", data=FinalDC)
plt.show()
#%%
def stories(row, colname): # colname can be 'rincome', 'income' etc
  thisstory = row[colname]
  if (thisstory < 20): return thisstory
  if (thisstory > 20): return np.nan
  return np.nan
# end function cleanDfIncome
print("\nReady to continue.")
# %%
FinalDC['STORIES'] = FinalDC.apply(stories, colname='STORIES', axis=1)
# %%
sns.regplot(x="STORIES", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"}, x_jitter=10, line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Stories")
plt.ylabel("Price")
plt.show()
# %%
sns.regplot(x="metro25", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"}, x_jitter=0,line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Categorical: 1 is Near Metro")
plt.ylabel("Price")
plt.show()
# %%
sns.regplot(x="metro50", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"},line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Categorical: 1 is Near Metro")
plt.ylabel("Price")
plt.show()
# %%
sns.regplot(x="LANDAREA", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"},line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Amount of Land Area")
plt.ylabel("Price")
plt.show()

# %%