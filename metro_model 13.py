
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
  lon1 = radians(point1[0])
  lon2 = radians(point2[0])
  lat1 = radians(point1[1])
  lat2 = radians(point2[1])
    
  # Haversine formula
  dlon = lon2 - lon1
  dlat = lat2 - lat1
  a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2

  c = 2 * asin(sqrt(a))
  
  # Radius of earth in kilometers. 
  # Use 3956 for miles
  # Use 6378.1 for km
  # Use 3958.8 for miles 
  # Use 6378100 for meters
  r = 3956
    
  # calculate the result
  return(c * r)
     
     
# driver code
# lat1 = 53.32055555555556
# lat2 = 53.31861111111111
# lon1 = -1.7297222222222221
# lon2 =  -1.6997222222222223
# print(distance(lat1, lat2, lon1, lon2), "K.M")
#%%
findClosestMetroDist( df.loc[0], list2 )
#%%
# df.apply(findClosestMetroDist)
# %%

#%%
df2 = [findClosestMetroDist(df.loc[i], list2) for i in range(0,12399)]
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
df4['distance', '.25metro', '.50metro'] = df3
df4.head()
# %%
print(df4)
# %%
df3.head()

# n = df3
# for i in n:
#       2*sqrt((n^2/2))

#%%
# def distance (row, dataframe):
#    homepoint = ( row['distance'])

#%%
def distance (hyp):
      h= hyp
      a= h**2
      b=a/2
      c=sqrt(b)
      d=2*c
      return d

# def distance (hyp)


# %%
def convert( row, frame ):
  pt = ( row['distance'] )
  
  result = [ distance( pt) for i in frame ]
  
  
  # for metrostation in metrostations:
  #   findDistBtw2Pts(metrostation[0],metrostation[1],x,y)
  return result



#%%
convert( df3.loc[2], df3 )
# %%
df6 = [convert( df3.loc[i], df3 ) for i in range(1,12398)]
# %%
print(df6)
# %%
column_names = ['distance']
df7 = pd.DataFrame(df6, columns=column_names)
# df3.rename(columns= {0:'distance'})
print(df7)
df7.describe()
# %%
plt.hist(x='distance', bins=20, data = df7)
# %%
totalDCdf= df4.copy()
totalDCdf.to_csv('FinalDCdf.csv')
#%%

df4['distance2'] = df7
secondDCdf=df4.copy()
secondDCdf.to_csv('newDistanceDC.csv')
# %%


#%%
def dummy25(row, colname): # colname can be 'rincome', 'income' etc
  thisdistance = row[colname]
  if (thisdistance <= .25): return 1
  if (thisdistance > .25): return 0
  return np.nan
# end function cleanDfIncome
print("\nReady to continue.")
# %%
df3['.25metro'] = df3.apply(dummy25, colname='distance', axis=1)
# %%
df3.head()

# %%
def dummy50(row, colname): # colname can be 'rincome', 'income' etc
  thisdistance = row[colname]
  if ( 0.25 < thisdistance <=0.5 ): return 1
  if (thisdistance > .5): return 0
  return np.nan
# end function cleanDfIncome
print("\nReady to continue.")
# %%
df3['.50metro'] = df3.apply(dummy25, colname='distance', axis=1)
#%%

def dummy1(row, colname): # colname can be 'rincome', 'income' etc
  thisdistance = row[colname]
  if (thisdistance <= 1): return 1
  if (thisdistance > 1): return 0
  return np.nan
# end function cleanDfIncome
print("\nReady to continue.")
# %%
df3['1metro'] = df3.apply(dummy1, colname='distance', axis=1)


#%%

def dummy2(row, colname): # colname can be 'rincome', 'income' etc
  thisdistance = row[colname]
  if (1 < thisdistance <=2): return 1
  if (thisdistance > 1): return 0
  return np.nan
# end function cleanDfIncome
print("\nReady to continue.")
# %%
df3['2metro'] = df3.apply(dummy2, colname='distance', axis=1)




# %%
df3
#%%

df4['distance'] = df3['distance']
#%%
df4['.25metro'] = df3['.25metro']

#%%
df4['.50metro'] = df3['.50metro']

df4['1metro'] = df3['1metro']

#%%
df4['2metro'] = df3['2metro']
df4.head()
# %%
totalDCdf= df4.copy()
totalDCdf.to_csv('FinalDC.csv')
# %%
