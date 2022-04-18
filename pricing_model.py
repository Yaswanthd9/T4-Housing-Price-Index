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