
#%%[markdown]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%

df= pd.read_csv('DC_Cleaned_Housing.csv')
df.head()

#%%
# metrostations = ( (-77.16462968, 39.11993515)  , (-77.14612767,39.08432943)  )  # (long,lat)
list = ( (-77.16462968, 39.11993515)  , (-77.14612767,39.08432943)  )

def findClosestMetroDist( row, metrostations ) :
  '''
  @row : a row of data in our dataframe
  return : distance (in meters) between the location and the nearest metro
  ''' 
  homepoint = ( row['X'] , row['Y'] )
  
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
findClosestMetroDist( df.loc[1], list )
#%%
# df.apply(findClosestMetroDist)
# %%

#%%
df2 = [findClosestMetroDist(df.loc[i], list) for i in range(1,12399)]
# %%
df2

# %%
