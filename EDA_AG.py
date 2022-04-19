#%%
import mysqlx
from mysqlx import Column
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print('Done, continue.')

#%%
FinalDC = pd.read_csv('/Users/Arundhati/Documents/T4-Housing-Price-Index/FinalDC.csv')
FinalDC.head()

# %%

# Creating a dummy column for EDA 
def DistanceDummy(distance): # colname can be 'rincome', 'income' etc
  
  if distance <= 0.25: return 1
  if distance > 0.25 and distance <= 0.50: return 2
  if distance > 0.5: return 3
  else: return 'NA'

#Creating the new column
FinalDC['DistanceDummy'] = FinalDC['distance'].apply(DistanceDummy)

#%%
def PRICE(PRICE): # colname can be 'rincome', 'income' etc
  PRICE = row[colname]
  if PRICE == 1: return np.nan
  if PRICE > 1: return PRICE
  else: return np.nan

FinalDC.dropna(inplace=True)


#Creating the new column
FinalDC['DistanceDummy'] = FinalDC['distance'].apply(DistanceDummy)


#Dropping 1 values in price
def PRICE(PRICE): # colname can be 'rincome', 'income' etc
  PRICE = row[colname]
  if PRICE == 1: return np.nan
  if PRICE > 1: return PRICE
  else: return np.nan

FinalDC.dropna(inplace=True)


#Creating the new column
FinalDC['DistanceDummy'] = FinalDC['distance'].apply(DistanceDummy)
# def PRICE(row, colname): # colname can be 'rincome', 'income' etc
#   thisprice = row[colname]
#   if thisprice == 1: return np.nan
#   if thisprice > 1: return thisprice
#   else: return np.nan

# FinalDC.dropna(inplace=True)


# #Creating the new column
# FinalDC['DistanceDummy'] = FinalDC['distance'].apply(DistanceDummy)


#%%
# display the dataframe
print(FinalDC)
  
#%%

#PLOTS FOR DISTANCE
# Overall observation: Price decreased as distance increased
#%%
#Violin Plot 
sns.violinplot(x="DistanceDummy", y="PRICE", data= FinalDC, scale="width")
plt.title("Price vs Distance")
plt.xticks(range(3), ['0.25', '0.50', 'Beyond 0.50'])
plt.xlabel("Distance to the metro")
plt.ylabel("Price")

plt.show()

#%%

# Joint Plot
sns.jointplot(x="distance", y="PRICE", data=FinalDC, color = 'blue', kind='reg', line_kws={'color':'green'})
plt.title("Price vs Distance")
plt.xlabel("Distance to the metro")
plt.ylabel("Price")
plt.savefig('DistanceJoint2.png')
plt.show()

x= ['0.5', '1', 'Greater than 1']
default_x_ticks = range(len(x))
plt.xticks(default_x_ticks, x)
plt.savefig('DistanceJoint.png')
plt.show()


#%%
#Regression Plot
sns.regplot(x="distance", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.title("Price vs Distance")
plt.xlabel("Distance to the metro")
plt.ylabel("Price")

plt.show()


#%%
#Scatter Plots
sns.scatterplot(x="distance", y="PRICE", data=FinalDC)
plt.show()
#%%
sns.scatterplot(x="DistanceDummy", y="PRICE", data=FinalDC)
plt.show()



# %%

#PLOTS FOR BEDROOMS
# Overall observation: price increased as the number of bedrooms increased

#Violin Plot 
sns.violinplot(x="BEDRM", y="PRICE", data= FinalDC, scale="width")
plt.title("Price vs Bed Rooms")
plt.xlabel("Number of Bed Rooms")
plt.ylabel("Price")

plt.show()

#%%

# Joint Plot
sns.jointplot(x="BEDRM", y="PRICE", data=FinalDC, color = 'blue', kind='reg', line_kws={'color':'green'})
plt.title("Price vs Bed Rooms")
plt.xlabel("Number of Bed Rooms")
plt.ylabel("Price")
plt.savefig('BedJoint.png')
plt.show()

#%%
#Regression Plot
sns.regplot(x="BEDRM", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.title("Price vs Bed Rooms")
plt.xlabel("Number of Bed Rooms")
plt.ylabel("Price")
plt.savefig('BedRegplot.png')
plt.show()

#%%
#Scatter Plots
sns.scatterplot(x="BEDRM", y="PRICE", data=FinalDC)
plt.savefig('BedroomScatter.png')
plt.show()


sns.scatterplot(x="BEDRM", y="PRICE", data=FinalDC)
plt.show()


# %%

#PLOTS FOR BATHROOMS
# Overall observation: price increased as the number of bedrooms increased

#Violin Plot 
sns.violinplot(x="BATHRM", y="PRICE", data= FinalDC, scale="width")
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.savefig('BathViolin.png')
plt.show()

#%%

# Joint Plot
sns.jointplot(x="BATHRM", y="PRICE", data=FinalDC, color = 'blue', kind='reg', line_kws={'color':'green'})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.savefig('BathJoint.png')
plt.show()
#%%

#Regression Plot
sns.regplot(x="BATHRM", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.savefig('BathRegplot.png')
plt.show()


# %%

#PLOTS FOR Half BATHROOMS
# Overall observation: price increased as the number of half baths increased
#%%
#Violin Plot 
sns.violinplot(x="HF_BATHRM", y="PRICE", data= FinalDC, scale="width")
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.savefig('halfViolin.png')
plt.show()


#%%
# Joint Plot
sns.jointplot(x="HF_BATHRM", y="PRICE", data=FinalDC, color = 'blue', kind='reg', line_kws={'color':'green'})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.savefig('hfJoint.png')
plt.show()

#%%
#Regression Plot
sns.regplot(x="HF_BATHRM", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.savefig('RoomRegplot.png')
plt.show()


# %%
#PLOTS FOR STORIES

FinalDC['STORIES'] = FinalDC['STORIES'].round()
FinalDC['STORIES'] = FinalDC['STORIES'].dropna()
print(FinalDC.STORIES)
#%%
#Violin Plot 
sns.violinplot(x="STORIES", y="PRICE", data= FinalDC, scale="width")
plt.title("Price vs Stories")
plt.xlabel("Number of Stories")
plt.ylabel("Price")
plt.savefig('StoriesViolin.png')
plt.show()

# %%
#PLOTS FOR AC

#Violin Plot 
sns.violinplot(x="AC", y="PRICE", data= FinalDC, scale="width")
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.savefig('BathViolin.png')
plt.show()


# %%

#PLOTS FOR Condition

#Violin Plot 
sns.violinplot(x="CNDTN", y="PRICE", data= FinalDC, scale="width")
plt.title("Price vs Condition")
plt.xlabel("Condition of the home")
plt.ylabel("Price")
plt.savefig('ConditionViolin.png')
plt.show()

#%%

# Plot for Land area
# #Violin Plot 
# sns.violinplot(x='LANDAREA', y="PRICE", data= FinalDC, scale="width")
# plt.title("Price vs Condition")
# plt.xlabel("Condition of the home")
# plt.ylabel("Price")
# plt.show()


# %%

import statsmodels.api as sm 
from statsmodels.formula.api import glm
# import stargazer
# from stargazer.stargazer import Stargazer
# from IPython.core.display import HTML

#%%
#renaming columns 
FinalDC['metro25'] = FinalDC['.25metro']
FinalDC['metro50'] = FinalDC['.50metro']
#%%
#GLM model with distance dummies 
glmmodel1 = glm(formula='PRICE ~ metro25 + metro50', data=FinalDC, family=sm.families.Binomial())

glmmodel1Fit = glmmodel1.fit()
print(glmmodel1Fit.summary())

# stargazer = Stargazer([glmmodel1Fit])
# HTML(stargazer.render_html())


#GLM model with all the variables 
glmmodel2 = glm(formula='PRICE ~ metro25 + metro50 + STORIES + LANDAREA + CNDTN + BATHRM + AC + LANDAREA', data=FinalDC, family=sm.families.Binomial())

glmmodel1Fit = glmmodel1.fit()
print( glmmodel1Fit.summary() )


#OLS to get the r-squared value 
from statsmodels.formula.api import ols

model1 = ols(formula='PRICE ~ STORIES + metro25 + metro50 + STORIES + LANDAREA + C(CNDTN) + BATHRM + BEDRM + AC + LANDAREA', data=FinalDC)

model1Fit = model1.fit()
print( model1Fit.summary() )
glmmodel1 = glm(formula='PRICE ~ metro50 + metro1', data=FinalDC, family=sm.families.Binomial())

glmmodel1Fit = glmmodel1.fit()
print(glmmodel1Fit.summary())
#%%
# stargazer = Stargazer([glmmodel1Fit])
# HTML(stargazer.render_html())

#%%
#GLM model with all the variables 
glmmodel2 = glm(formula='PRICE ~ metro50 + metro1 + STORIES + LANDAREA + CNDTN + BATHRM + HF_BATHRM + AC', data=FinalDC)

glmmodel2Fit = glmmodel2.fit()
print( glmmodel2Fit.summary() )

#%%
#OLS to get the r-squared value 
from statsmodels.formula.api import ols
#%%
model3 = ols(formula='PRICE ~ metro50 + metro1 + STORIES + LANDAREA + CNDTN + BATHRM + HF_BATHRM + AC', data=FinalDC)

model3Fit = model3.fit()
print( model3Fit.summary() )



#Logging Price 
FinalDC['log_price'] = np.log2(FinalDC['PRICE'])

model1 = ols(formula='log_price ~ metro25 + metro50 + STORIES + LANDAREA + C(CNDTN) + BATHRM + BEDRM + AC + LANDAREA', data=FinalDC)
model1Fit = model1.fit()
print(model1Fit.summary())


#Violin Plot 
sns.violinplot(x="CNDTN", y="PRICE", data= FinalDC, scale="width")
plt.title("Price vs Condition")
plt.xlabel("Condition of the home")
plt.ylabel("Price")
plt.show()


#Violin Plot 
sns.violinplot(x='LANDAREA', y="PRICE", data= FinalDC, scale="width")
plt.title("Price vs Condition")
plt.xlabel("Condition of the home")
plt.ylabel("Price")
plt.show()


# %%
modelsurvivalLogit = glm(formula='PRICE ~ distance + STORIES + LANDAREA + CNDTN +   ', data=FinalDC, family=sm.families.Binomial())

#renaming columns 
FinalDC['metro25'] = FinalDC['.25metro']
FinalDC['metro50'] = FinalDC['.50metro']

#GLM model with distance dummies 
glmmodel1 = glm(formula='PRICE ~ metro25 + metro50', data=FinalDC, family=sm.families.Binomial())

glmmodel1Fit = glmmodel1.fit()
print( glmmodel1Fit.summary() )


#GLM model with all the variables 
glmmodel2 = glm(formula='PRICE ~ metro25 + metro50 + STORIES + LANDAREA + CNDTN + BATHRM + AC + LANDAREA', data=FinalDC, family=sm.families.Binomial())

glmmodel1Fit = glmmodel2.fit()
print( glmmodel1Fit.summary() )


#OLS to get the r-squared value 
from statsmodels.formula.api import ols

model3 = ols(formula='PRICE ~ metro25 + metro50 + STORIES + LANDAREA + CNDTN + BATHRM + HF_BATHRM + AC', data=FinalDC)

model3Fit = model3.fit()
print( model3Fit.summary() )


#Logging Price 
FinalDC['log_price'] = np.log2(FinalDC['PRICE'])

model4 = ols(formula='log_price ~ metro25 + metro50 + STORIES + LANDAREA + CNDTN + BATHRM + HF_BATHRM + AC', data=FinalDC)
model4Fit = model4.fit()
print(model4Fit.summary())




model4 = ols(formula='log_price ~ metro25 + metro50 + STORIES + LANDAREA + CNDTN + BATHRM + HF_BATHRM + AC', data=FinalDC)
model4Fit = model4.fit()
print(model4Fit.summary())






# %%

#### OTHER SCATTER PLOTS ####
#Regression Plot
sns.regplot(x="BATHRM", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"}, x_jitter=10, line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.savefig('BathRegplot.png')
plt.show()
# %%
sns.regplot(x="ROOMS", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"}, x_jitter=10, line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Rooms")
plt.ylabel("Price")
plt.savefig('RoomRegplot.png')
plt.show()
#%%
#%%
sns.scatterplot(x="distance", y="PRICE", data=FinalDC)
plt.savefig('priceVsDistance.png')
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
plt.savefig('priceVsstories.png')
plt.show()
# %%
sns.regplot(x="metro25", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"}, x_jitter=10,line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Categorical: 1 is Near Metro")
plt.ylabel("Price")
plt.savefig('priceVsMetro50.png')
plt.show()
# %%
sns.regplot(x="metro50", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"},line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Categorical: 1 is Near Metro")
plt.ylabel("Price")
plt.savefig('priceVsMetro25.png')
plt.show()
# %%
sns.regplot(x="LANDAREA", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"},line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Amount of Land Area")
plt.ylabel("Price")
plt.savefig('priceVsLand.png')
plt.show()

# %%
sns.lmplot(x="LANDAREA", y="PRICE", hue = 'DistanceDummy', data=FinalDC, fit_reg = True, x_jitter=10)
plt.title("Price vs Bath Rooms")
plt.xlabel("Amount of Land Area")
plt.ylabel("Price")
plt.savefig('priceVsLand.png')
plt.show()
# %%
print(FinalDC.metro50.describe())
#%%
