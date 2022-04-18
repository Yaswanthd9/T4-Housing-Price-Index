#%%
import mysqlx
from mysqlx import Column
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print('Done, continue.')

#%%
FinalDC = pd.read_csv('FinalDC.csv')
FinalDC.head()

# %%

def DistanceDummy(distance): # colname can be 'rincome', 'income' etc
  
  if distance <= 0.25: return 1
  if distance > 0.25 and distance <= 0.50: return 2
  if distance > 0.5: return 3
  else: return 'NA'


#Creating the new column
FinalDC['DistanceDummy'] = FinalDC['distance'].apply(DistanceDummy)


def PRICE(PRICE): # colname can be 'rincome', 'income' etc
  PRICE = row[colname]
  if PRICE == 1: return np.nan
  if PRICE > 1: return PRICE
  else: return np.nan

FinalDC.dropna(inplace=True)


#Creating the new column
FinalDC['DistanceDummy'] = FinalDC['distance'].apply(DistanceDummy)



# display the dataframe
print(FinalDC)
  
#%%

#PLOTS FOR DISTANCE
# Overall observation: Price decreased as distance increased

#Violin Plot 
sns.violinplot(x="DistanceDummy", y="PRICE", data= FinalDC, scale="width")
plt.title("Price vs Distance")
plt.xticks(range(3), ['0.25', '0.50', 'Beyond 0.50'])
plt.xlabel("Distance to the metro")
plt.ylabel("Price")
plt.show()



# Joint Plot
sns.jointplot(x="distance", y="PRICE", data=FinalDC, color = 'blue', kind='reg', line_kws={'color':'green'})
plt.title("Price vs Distance")
plt.xlabel("Distance to the metro")
plt.ylabel("Price")
plt.show()


#Regression Plot
sns.regplot(x="distance", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.title("Price vs Distance")
plt.xlabel("Distance to the metro")
plt.ylabel("Price")
plt.show()



#Scatter Plots
sns.scatterplot(x="distance", y="PRICE", data=FinalDC)
plt.show()

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



# Joint Plot
sns.jointplot(x="BEDRM", y="PRICE", data=FinalDC, color = 'blue', kind='reg', line_kws={'color':'green'})
plt.title("Price vs Bed Rooms")
plt.xlabel("Number of Bed Rooms")
plt.ylabel("Price")
plt.show()


#Regression Plot
sns.regplot(x="BEDRM", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.title("Price vs Bed Rooms")
plt.xlabel("Number of Bed Rooms")
plt.ylabel("Price")
plt.show()


#Scatter Plots
sns.scatterplot(x="BEDRM", y="PRICE", data=FinalDC)
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
plt.show()



# Joint Plot
sns.jointplot(x="BATHRM", y="PRICE", data=FinalDC, color = 'blue', kind='reg', line_kws={'color':'green'})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.show()


#Regression Plot
sns.regplot(x="BATHRM", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.show()


# %%

#PLOTS FOR Half BATHROOMS
# Overall observation: price increased as the number of half baths increased

#Violin Plot 
sns.violinplot(x="HF_BATHRM", y="PRICE", data= FinalDC, scale="width")
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.show()



# Joint Plot
sns.jointplot(x="HF_BATHRM", y="PRICE", data=FinalDC, color = 'blue', kind='reg', line_kws={'color':'green'})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.show()


#Regression Plot
sns.regplot(x="HF_BATHRM", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.show()


# %%
#PLOTS FOR STORIES

FinalDC['STORIES'] = FinalDC['STORIES'].round()
FinalDC['STORIES'] = FinalDC['STORIES'].dropna()
print(FinalDC.STORIES)

#Violin Plot 
sns.violinplot(x="STORIES", y="PRICE", data= FinalDC, scale="width")
plt.title("Price vs Stories")
plt.xlabel("Number of Stories")
plt.ylabel("Price")
plt.show()

# %%
#PLOTS FOR AC

#Violin Plot 
sns.violinplot(x="AC", y="PRICE", data= FinalDC, scale="width")
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.show()


# %%

#PLOTS FOR Condition

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

import statsmodels.api as sm 
from statsmodels.formula.api import glm

modelsurvivalLogit = glm(formula='PRICE ~ distance + STORIES + LANDAREA + CNDTN +   ', data=FinalDC, family=sm.families.Binomial())

modelsurvivalLogitFit = modelsurvivalLogit.fit()
print( modelsurvivalLogitFit.summary() )



