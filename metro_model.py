
#%%[markdown]
from mysqlx import Column
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%

df= pd.read_csv('DC_Cleaned_Housing.csv')
df.head()
#%%
metro = pd.read_csv('Metro_Stations_Regional.csv')
metro.head()
metro_new = metro.drop(['WEB_URL', 'TRAININFO_URL','SE_ANNO_CAD_DATA', 'OBJECTID', 'CREATED', 'EDITOR','EDITED', 'CREATOR' ], axis=1)
metro_new.head()
#%%
# metro_new.to_csv('Metro_Cleaned.csv')
df_new = metro_new.iloc[:, [0,1,]]
df_new.head()
#%%
print(df_new)
#%%
records= df_new.to_records(index=False)
print(records)
#%%


#%%
# metrostations = ( (-77.16462968, 39.11993515)  , (-77.14612767,39.08432943)  )  # (long,lat)
# list2 = ( (-77.16462968, 39.11993515)  , (-77.14612767,39.08432943)  )
list2 = records
#%%
def findClosestMetroDist( row, metrostations ) :
  '''
  @row : a row of data in our dataframe
  return : distance (in meters) between the location and the nearest metro
  ''' 
  homepoint = ( row['LONGITUDE'] , row['LATITUDE'] )
  
  result = [ findDistBtw2Pts( metrostation , homepoint) for metrostation in metrostations ]
  
  result = min(result)
  # for metrostation in metrostations:
  #   findDistBtw2Pts(metrostation[0],metrostation[1],x,y)
  return result
  
# def findDistBtw2Pts( point1 ,point2 ):
#   dist=0
#   return dist


# Python 3 program to calculate Distance Between Two Points on Earth
from math import radians, cos, sin, asin, sqrt
def findDistBtw2Pts( point1 ,point2 ):
  # The math module contains a function named
  # radians which converts from degrees to radians.
  lon1 = radians(point1[1])
  lon2 = radians(point2[1])
  lat1 = radians(point1[0])
  lat2 = radians(point1[0])
    
  # Haversine formula
  dlon = lon2 - lon1
  dlat = lat2 - lat1
  a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2

  c = 2 * asin(sqrt(a))
  
  # Radius of earth in kilometers. Use 3956 for miles
  r = 6378100
    
  # calculate the result
  return(c * r)
     
     
# driver code
# lat1 = 53.32055555555556
# lat2 = 53.31861111111111
# lon1 = -1.7297222222222221
# lon2 =  -1.6997222222222223
# print(distance(lat1, lat2, lon1, lon2), "K.M")
#%%
findClosestMetroDist( df.loc[1], list2 )
#%%
# df.apply(findClosestMetroDist)
# %%

#%%
df2 = [findClosestMetroDist(df.loc[i], list2) for i in range(1,12399)]
# %%
#this is a list still
df2
#%%
column_names = ['distance']
df3 = pd.DataFrame(df2, columns=column_names)
# df3.rename(columns= {0:'distance'})
print(df3)
df3.describe()
# %%
plt.hist(x='distance', bins=50, data = df3)

#%%
df4 = df.copy()
df4.describe()
# %%
df4['distance'] = df3
df4.head()
# %%
print(df4)
# %%

