
#%%
listA=  ( (-99.436554, 41.507483))

listB=  ( (-98.315949, 38.504048))


#%%
import numpy as np
import math as m
# from math import atan2, radians, cos, sin, asin, sqrt
def findDistBtw2Pts( point1, point2):
  # The math module contains a function named
  # radians which converts from degrees to radians.
  lon1 = m.radians(point1[0])
  lon2 = m.radians(point2[0])
  lat1 = m.radians(point1[1])
  lat2 = m.radians(point2[1])
    
  # Haversine formula
  dlon = lon2 - lon1
  dlat = lat2 - lat1
  a = m.sin(dlat / 2)**2 + m.cos(lat1) * m.cos(lat2) * m.sin(dlon / 2)**2

  c = 2 * m.atan2(m.sqrt(a), m.sqrt(1-a))
  
  # Radius of earth in kilometers. 
  # Use 3956 for miles
  # Use 6378.1 for km
  # Use 3958.8 for miles 
  # Use 6378100 for meters
  r = 6378.1
    
  # calculate the result
  return(c * r)


#%%
findDistBtw2Pts( listA, listB )
# %%
