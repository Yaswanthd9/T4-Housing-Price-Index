#%%[markdown]
from mysqlx import Column
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#%%

# %%
dc = pd.read_csv('DCHousing.csv')
# %%
dc.head()
#%%
## Dropping the columns from the dataset that we do not need:
dc_new = dc.drop(['HEAT', 'AYB', 'YR_RMDL', 'QUALIFIED', 'SALE_NUM', 'GBA', 'BLDG_NUM', 'STYLE', 'GRADE', 'EXTWALL', 'ROOF', 'INTWALL', 'USECODE', 'GIS_LAST_MOD_DTTM', 'SOURCE', 'CMPLX_NUM', 'LIVING_GBA', 'ASSESSMENT_NBHD', 'CENSUS_TRACT', 'CENSUS_BLOCK'], axis=1)
dc_new.head()

dc_new['year']= dc.SALEDATE.astype(str).str[:4]
dc_new.head()
dc_new= dc_new.drop(['SALEDATE'], axis=1)

dc_new.year.describe()

# %%
metro = pd.read_csv('Metro_Stations_Regional.csv')
metro.head()
metro_new = metro.drop(['WEB_URL', 'TRAININFO_URL','SE_ANNO_CAD_DATA', 'OBJECTID', 'CREATED', 'EDITOR','EDITED', 'CREATOR' ], axis=1)
metro_new.head()
# %%
#%%
# metro_new.to_csv('Metro_Cleaned.csv')
df_new = metro_new.iloc[:, [0,1,]]
df_new.head()

print(df_new)
# %%
records= df_new.to_records(index=False)
print(records)

#%%
# metrostations = ( (-77.16462968, 39.11993515)  , (-77.14612767,39.08432943)  )  # (long,lat)
# list2 = ( (-77.16462968, 39.11993515)  , (-77.14612767,39.08432943)  )
list2 = records

#%%

plt.hist(dc_new.BEDRM)
plt.show() 
plt.xlabel('Histograms for Bedrooms')


# %%
import plotly.express as px
fig = px.histogram(dc_new.BEDRM, x="BEDRM")
fig.show()
# %%
import plotly.express as px
fig = px.histogram(dc_new.PRICE, x="PRICE",nbins=1)
fig.show()
# %%
import plotly.express as px
fig = px.histogram(dc_new.BATHRM, x="BATHRM")
fig.show()

# %%
import plotly.express as px
fig = px.histogram(dc_new.HF_BATHRM, x="HF_BATHRM")
fig.show()
# %%
import plotly.express as px
fig = px.histogram(dc_new.ROOMS, x="ROOMS")
fig.show()
# %%
import plotly.express as px
fig = px.histogram(dc_new.EYB, x="EYB")
fig.show()
# %%
import plotly.express as px
fig = px.histogram(dc_new.STORIES, x="STORIES",nbins=1)
fig.show()
# %%
import plotly.express as px
fig = px.histogram(dc_new.year, x="year")
fig.show()
# %%
import plotly.express as px
fig = px.histogram(dc_new.CNDTN, x="CNDTN")
fig.show()
# %%
import plotly.express as px
fig = px.histogram(dc_new.KITCHENS, x="KITCHENS")
fig.show()
# %%

import plotly.express as px
fig = px.histogram(dc_new.SQUARE, x="SQUARE")
fig.show()
# %%
