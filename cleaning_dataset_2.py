
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import dm6103 as dm
# %%

dc = pd.read_csv('DCHousing.csv')
# %%
dc.head()
#%%
## Dropping the columns from the dataset that we do not need:
dc_new = dc.drop(['HEAT', 'AYB', 'YR_RMDL', 'QUALIFIED', 'SALE_NUM', 'GBA', 'BLDG_NUM', 'STYLE', 'GRADE', 'EXTWALL', 'ROOF', 'INTWALL', 'USECODE', 'GIS_LAST_MOD_DTTM', 'SOURCE', 'CMPLX_NUM', 'LIVING_GBA', 'ASSESSMENT_NBHD', 'CENSUS_TRACT', 'CENSUS_BLOCK'], axis=1)
dc_new.head()
#%%
## Seperating out the SALEDATe column
dc_new['year']= dc.SALEDATE.astype(str).str[:4]
dc_new.head()
dc_new= dc_new.drop(['SALEDATE'], axis=1)
#%%
dc_new.year.describe()
#%%
dc_2017 = (dc_new['year'] == '2017')
dc_2017.head()
# print(dc_2017)
dc_new_2017= dc_new[dc_2017]
# dc_new_2017.head()
dc_new_2017.describe()
# %%
dc_new_2017.to_csv('DC_Cleaned_Housing.csv')



#####################
#%%
metro = pd.read_csv('Metro_Stations_Regional.csv')
metro.head()
metro_new = metro.drop(['WEB_URL', 'TRAININFO_URL','SE_ANNO_CAD_DATA', 'OBJECTID', 'CREATED', 'EDITOR','EDITED', 'CREATOR' ], axis=1)

metro_new.head()

metro_new.to_csv('Metro_Cleaned.csv')





#%%
print(metro_new)
#%%

df_new = metro_new.iloc[:, [0,1,]]
df_new.head()
#%%
 
# def toTuple( row, df_new ) :
#       '''
#   @row : a row of data in our dataframe
#   return : distance (in meters) between the location and the nearest metro
#   ''' 
#   points = ( row['X'] , row['Y'] )

#   result = ( (homepoint) for row in df_new )
  
#   return result



#%%
print(df_new)
records= df_new.to_records(index=False)
list1 = list(records)
print(list1)







#%%

df_new
metro_new['lat_long'] = metro_new[['X', 'Y']].apply(tuple, axis=1)
metro = metro_new['lat_long']

#%%
print(metro_new['lat_long'])

#%%
list2 = [ (metro_new['lat_long'][1]) for i in metro_new['lat_long'] ]

print(list2)

#%%

tuple1 = tuple(zip(metro, metro))
#%%
print(tuple)
#%%
tuple2 = tuple(zip(metro.items()))
#%%
metro.items() or metro.iteritems()
#%%
tuple = tuple(zip(metro.items()))
tuple = tuple(zip(metro.items()))
tuple = tuple(zip(metro.items()))
print(tuple)




















########### SCRATCH WORK ##########
###################################
#%%
meh= pd.read_csv('MEHOINUSA672N.csv')
meh.head()
# %%
dc_2017 = dc['AYB']==2017
# %%
dc_2017.describe()
# %%
