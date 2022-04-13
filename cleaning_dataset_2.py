
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
