#%%[markdown]
from mysqlx import Column
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeClassifier

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
from mysqlx import Column
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm 
from statsmodels.formula.api import ols
from statsmodels.formula.api import glm
from statsmodels.formula.api import logit
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model, metrics
fatcalfit = LogisticRegression(max_iter= 1000)
from sklearn.model_selection import train_test_split
#%%
## Import finalized dataset as .csv##
pricedf1= pd.read_csv('FinalDC.csv')
# %%f
pricedf1.head()
# %%
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

feature_cols=['.25metro','.50metro','BEDRM','SQUARE','BATHRM','distance','SQUARE','ROOMS','STORIES']
X = pricedf1[feature_cols] # Features
y = pricedf1.PRICE # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# %%
FinalDC = pd.read_csv('FinalDC.csv')
FinalDC.head()
sns.regplot(x="HF_BATHRM", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.savefig('RoomRegplot.png')
plt.show()
# %%
FinalDC['log_price'] = np.log2(FinalDC['PRICE'])
plt.hist(x='log_price',bins=80, data= FinalDC)
plt.xlabel('Price, 1e7')
plt.ylabel('Frequency')
plt.title('Distribution of Home Prices')
plt.savefig('LogpriceHist.png')
plt.show()
# %%
FinalDC['log_price'] = np.log2(FinalDC['PRICE'])
plt.hist(x='log_price',bins=80, data= FinalDC)
plt.xlabel('log Price')
plt.ylabel('Frequency')
plt.title('Distribution of Home Prices')
plt.axvline(FinalDC.log_price.median(), color='red', linestyle='dashed', linewidth=1)
plt.axvline(FinalDC.log_price.mean(), color='black', linestyle='dashed', linewidth=1)
plt.savefig('LogPriceHist.png')
plt.show()
# %%
