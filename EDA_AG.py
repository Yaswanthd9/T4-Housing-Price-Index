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

#PLOTS FOR Bathrooms
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
#PLOTS FOR Bathrooms
# Overall observation: price increased as the number of bedrooms increased

FinalDC['AC'] = 

#Violin Plot 
sns.violinplot(x="AC", y="PRICE", data= FinalDC, scale="width")
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.show()


# %%
