#%%
import mysqlx
from mysqlx import Column
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
print('Done, continue.')
#%%
###################################
### Load in the Cleaned Dataset ###
###################################
FinalDC = pd.read_csv('FinalDC.csv')
FinalDC.head()
# %%
###################################
# Creating a dummy column for 3 
# categories of distance for EDA
# ################################### 
def DistanceDummy(distance): 
  if distance <= 0.50: return 1
  if distance > 0.50 and distance <= 1: return 2
  if distance > 1: return 3
  else: return 'NA'
FinalDC['DistanceDummy'] = FinalDC['distance'].apply(DistanceDummy)
#%%
#############################
####### PRICE HISTOGRAM #####
#############################
plt.hist(x='PRICE',bins=80, data= FinalDC)
plt.xlabel('Price, 1e7')
plt.ylabel('Frequency')
plt.title('Distribution of Home Prices')
plt.savefig('priceHist.png')
plt.show()
#%%
#############################
##### LOG PRICE HISTOGRAM ###
#############################
FinalDC['log_price'] = np.log2(FinalDC['PRICE'])
plt.hist(x='log_price',bins=80, data= FinalDC)
plt.xlabel('Price, 1e7')
plt.ylabel('Frequency')
plt.title('Distribution of Home Prices')
plt.savefig('LogpriceHist.png')
plt.show()
#%%
outBig= (8.150000e+05+(1.5*(8.150000e+05-3.650000e+05)))
outSmall = (3.650000e+05-(1.5*(8.150000e+05-3.650000e+05)))
#%%
def priceOutlier(row, colname): 
  thisprice = row[colname]
  if (3000 < thisprice < outBig ): return thisprice
  if (outBig < thisprice < outSmall ): return np.nan
  return np.nan
print("\nReady to continue.")
FinalDC['newPrice'] = FinalDC.apply(priceOutlier, colname='PRICE', axis=1)
print("\nReady to continue.")
#%%
#############################
##### NEW PRICE HISTOGRAM ###
#############################
plt.hist(x='newPrice',bins=80, data= FinalDC)
plt.xlabel('Price, 1e6')
plt.ylabel('Frequency')
plt.title('Distribution of Home Prices')
plt.axvline(FinalDC.newPrice.median(), color='red', linestyle='dashed', linewidth=1)
plt.axvline(FinalDC.newPrice.mean(), color='black', linestyle='dashed', linewidth=1)
plt.savefig('newPriceHist.png')
plt.show()
#%%
#############################
##### LOG PRICE HISTOGRAM ###
#############################
FinalDC['newlog_price'] = np.log2(FinalDC['newPrice'])
plt.hist(x='newlog_price',bins=80, data= FinalDC)
plt.xlabel('log Price')
plt.ylabel('Frequency')
plt.title('Distribution of Home Prices')
plt.axvline(FinalDC.newlog_price.median(), color='red', linestyle='dashed', linewidth=1)
plt.axvline(FinalDC.newlog_price.mean(), color='black', linestyle='dashed', linewidth=1)
plt.savefig('newLogPriceHist.png')
plt.show()
#%%
#############################
#### DISTANCE HISTOGRAM #####
#############################
plt.hist(x='distance',bins=50, data= FinalDC)
plt.xlabel('Distance (mi)')
plt.ylabel('Frequency')
plt.title('Distribution of Distances')
plt.axvline(FinalDC.distance.median(), color='red', linestyle='dashed', linewidth=1)
plt.savefig('distanceHist.png')
plt.show()
#%%
#############################
##### BEDROOM HISTOGRAM #####
#############################
plt.hist(x='BEDRM',bins=15, data= FinalDC)
plt.xlabel('Number of Bedrooms')
plt.ylabel('Frequency')
plt.title('Distribution of Bedrooms')
plt.axvline(FinalDC.BEDRM.median(), color='red', linestyle='dashed', linewidth=1)
plt.savefig('bedroomHist.png')
plt.show()
#%%
#############################
####### ROOMS HISTOGRAM #####
#############################
plt.hist(x='ROOMS',bins=15, data= FinalDC)
plt.xlabel('Number of Rooms')
plt.ylabel('Frequency')
plt.title('Distribution of Rooms')
plt.axvline(FinalDC.ROOMS.median(), color='red', linestyle='dashed', linewidth=1)
plt.savefig('roomsHist.png')
plt.show()
#%%
#############################
##### STRUCTURE HISTOGRAM ###
#############################
def structure(row, colname): # colname can be 'rincome', 'income' etc
  thisstructure = row[colname]
  if (thisstructure == 'Row Inside'): return 1
  if (thisstructure == 'Multi' ): return 2
  if (thisstructure == 'Semi-Detached' ): return 3
  if (thisstructure == 'Row End' ): return 4
  if (thisstructure == 'Single' ): return 5
  if (thisstructure == 'Town End' ): return np.nan
  if (thisstructure == 'Town Inside' ): return np.nan
  return np.nan
print("\nReady to continue.")
FinalDC['structure'] = FinalDC.apply(structure, colname='STRUCT', axis=1)
#%%
plt.hist(x='structure',bins=15, data= FinalDC)
plt.xlabel('Type of Home')
plt.ylabel('Frequency')
plt.title('Distribution of Home Type')
x= ['','Row Inside', 'Multi', 'Semi-Det.', 'Row-End', 'Single']
default_x_ticks = range(len(x))
plt.xticks(default_x_ticks, x)
# plt.axvline(FinalDC.ROOMS.median(), color='red', linestyle='dashed', linewidth=1)
plt.savefig('structureHist.png')
plt.show()
#%%
#############################
##### DISTANCE VIOLIN #######
#############################
sns.violinplot(x="DistanceDummy", y="newPrice",data= FinalDC, scale="width")
plt.title("Price vs Distance")
plt.xticks(range(3), ['0.25', '0.50', 'Beyond 0.50'])
plt.xlabel("Distance to the metro")
plt.ylabel("Price")
plt.show()
#%%
#############################
##### DISTANCE JOINTPLOT ####
#############################
sns.jointplot(x="distance", y="newPrice", data=FinalDC, color = 'blue', kind='reg', line_kws={'color':'red'})
plt.suptitle("Price vs Distance")
# plt.title("Price vs Distance")
# plt.xlabel("Distance to the metro")
# sns.ylabel("Price")
# x= ['0.5', '1', 'Greater than 1']
# default_x_ticks = range(len(x))
# plt.xticks(default_x_ticks, x)
plt.savefig('DistanceJoint.png')
plt.show()
# %%
#############################
##### BEDROOM VIOLIN ########
#############################

# Overall observation: price increased as the number of bedrooms increased

sns.violinplot(x="BEDRM", y="newPrice", data= FinalDC, scale="width")
plt.title("Price vs Bed Rooms")
plt.xlabel("Number of Bed Rooms")
plt.ylabel("Price")
plt.savefig('BedViolin.png')
plt.show()
#%%
#############################
##### BEDROOM JOINTPLOT #####
#############################
sns.jointplot(x="BEDRM", y="newPrice", data=FinalDC, color = 'blue', kind='reg', line_kws={'color':'green'})
plt.title("Price vs Bed Rooms")
plt.xlabel("Number of Bed Rooms")
plt.ylabel("Price")
plt.savefig('BedJoint.png')
plt.show()

# %%
#############################
##### BATHROOM VIOLIN #######
#############################
# Overall observation: price increased as the number of bedrooms increased

sns.violinplot(x="BATHRM", y="newPrice", data= FinalDC, scale="width")
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.savefig('BathViolin.png')
plt.show()
#%%
#############################
#### BATHROOM JOINTPLOT #####
#############################
sns.jointplot(x="BATHRM", y="newPrice", data=FinalDC, color = 'blue', kind='reg', line_kws={'color':'green'})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.savefig('BathJoint.png')
plt.show()
# %%
#PLOTS FOR Half BATHROOMS
# Overall observation: price 
# increased as the number of half 
# baths increased
#%%
#############################
##### HALFBATH VIOLIN #######
#############################
sns.violinplot(x="HF_BATHRM", y="newPrice", data= FinalDC, scale="width")
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.savefig('halfViolin.png')
plt.show()
#%%
#############################
#### HALFBATH JOINTPLOT #####
#############################
sns.jointplot(x="HF_BATHRM", y="newPrice", data=FinalDC, color = 'blue', kind='reg', line_kws={'color':'green'})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Half-BathRooms")
plt.ylabel("Price")
plt.savefig('hfJoint.png')
plt.show()
# %%
#PLOTS FOR STORIES
FinalDC['STORIES'] = FinalDC['STORIES'].round()
FinalDC['STORIES'] = FinalDC['STORIES'].dropna()
print(FinalDC.STORIES)
#%%
#############################
###### STORIES VIOLIN #######
############################# 
sns.violinplot(x="STORIES", y="newPrice", data= FinalDC, scale="width")
plt.title("Price vs Stories")
plt.xlabel("Number of Stories")
plt.ylabel("Price")
plt.savefig('StoriesViolin.png')
plt.show()

# %%
#############################
######### AC VIOLIN #########
############################# 
sns.violinplot(x="AC", y="newPrice", data= FinalDC, scale="width")
plt.title("Price vs Bath Rooms")
plt.xlabel("Air Conditioner")
plt.ylabel("Price")
plt.savefig('acViolin.png')
plt.show()


# %%

#PLOTS FOR Condition

#Violin Plot 
sns.violinplot(x="CNDTN", y="newPrice", data= FinalDC, scale="width")
plt.title("Price vs Condition")
plt.xlabel("Condition of the home")
plt.ylabel("Price")
plt.savefig('ConditionViolin.png')
plt.show()

#%%

# Plot for Land area
# #Violin Plot 
# sns.violinplot(x='LANDAREA', y="PRICE", data= FinalDC, scale="width")
# plt.title("Price vs Condition")
# plt.xlabel("Condition of the home")
# plt.ylabel("Price")
# plt.show()


# %%

import statsmodels.api as sm 
from statsmodels.formula.api import glm
from statsmodels.formula.api import ols
# import stargazer
# from stargazer.stargazer import Stargazer
# from IPython.core.display import HTML

#%%
#renaming columns 
# FinalDC['metro25'] = FinalDC['.25metro']
# FinalDC['metro50'] = FinalDC['.50metro']
#%%
#GLM model with distance dummies 
glmmodel1 = ols(formula='PRICE ~ metro50 ', data=FinalDC)

glmmodel1Fit = glmmodel1.fit()
print(glmmodel1Fit.summary())
#%%
#
#  stargazer = Stargazer([glmmodel1Fit])
# HTML(stargazer.render_html())

#%%
#GLM model with all the variables 
glmmodel2 = glm(formula='PRICE ~ metro50  + STORIES + LANDAREA + CNDTN + BATHRM + HF_BATHRM + AC', data=FinalDC)

glmmodel2Fit = glmmodel2.fit()
print( glmmodel2Fit.summary() )

#%%
#OLS to get the r-squared value 
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
#%%
model3 = ols(formula='PRICE ~ metro50  + STORIES + LANDAREA + CNDTN + BATHRM + HF_BATHRM + AC', data=FinalDC)

model3Fit = model3.fit()
print( model3Fit.summary() )

#%%
#Logging Price 
FinalDC['log_price'] = np.log2(FinalDC['PRICE'])
#%%
dfCorr = pd.DataFrame(FinalDC, columns= ['log_price', 'CNDTN', 'AC', 'metro50', 'STORIES', 'LANDAREA', 'BATHRM', 'ROOMS', 'HF_BATHRM'])
#%%
correlation = dfCorr.corr()
print(correlation)
#%%
correlation.to_csv('corrMatrix.csv')

#%%
model4 = ols(formula='log_price ~ metro50 + STORIES + LANDAREA + CNDTN + BATHRM + HF_BATHRM + AC', data=FinalDC)
model4Fit = model4.fit(cov_type='HC3')
print(model4Fit.summary())

#%%
model5 = ols(formula='log_price ~ metro50', data=FinalDC)
model5Fit = model5.fit(cov_type='HC3')
print(model5Fit.summary())
#%%
model6 = ols(formula='log_price ~ metro50 + ROOMS + BATHRM + CNDTN + HF_BATHRM', data=FinalDC)
model6Fit = model6.fit(cov_type='HC3')
print(model6Fit.summary())
#%%
FinalDC['roomsSQ']= FinalDC['ROOMS']*FinalDC['ROOMS']
FinalDC['roomsSQ'].head()
#%%
FinalDC['bathSQ']= FinalDC['BATHRM']*FinalDC['BATHRM']
FinalDC['bathSQ'].head()
#%%
FinalDC['bedSQ']= FinalDC['BEDRM']*FinalDC['BEDRM']
FinalDC['bedSQ'].head()


#%%
formulaI= 'log_price ~ bedSQ+roomsSQ+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM'
model7 = ols(formula=formulaI , data=FinalDC)
model7Fit = model7.fit(cov_type='HC3')
print(model7Fit.summary())
#%%
formulaI= 'newlog_price ~ bedSQ+roomsSQ+metro50+ROOMS+HF_BATHRM+BATHRM +C(structure) '
model7 = ols(formula=formulaI , data=FinalDC)
model7Fit = model7.fit(cov_type='HC3')
print(model7Fit.summary())

#%%
FinalDC.head()

# %%

#%%
model_fitted_y =model4Fit.fittedvalues
model_residuals = model4Fit.resid

plt.scatter(x=model_residuals, y= model_fitted_y)
plt.show()
#%%
model_norm_residuals = model4Fit.get_influence().resid_studentized_internal
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
model_abs_resid = np.abs(model_residuals)
model_leverage = model4Fit.get_influence().hat_matrix_diag
plot_lm_1 = plt.figure()
plot_lm_1.axes[0] = sns.residplot(model_abs_resid,model_fitted_y, data=FinalDC)
plt.title('Residuals vs Fitted')
plt. xlabel('Fitted values')
plt.ylabel('Residuals')
plt.savefig('resid')

#%%
from statsmodels.graphics.gofplots import ProbPlot
#%%
QQ = ProbPlot(model_norm_residuals)
plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
plt.title('Normal Q-Q')
plt.savefig('normalqq')
# %%

#### OTHER SCATTER PLOTS ####
#Regression Plot
sns.regplot(x="BATHRM", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"}, x_jitter=10, line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.savefig('BathRegplot.png')
plt.show()
# %%
sns.regplot(x="ROOMS", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"}, x_jitter=10, line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Rooms")
plt.ylabel("Price")
plt.savefig('RoomRegplot.png')
plt.show()
#%%
#%%
sns.scatterplot(x="distance", y="PRICE", data=FinalDC)
plt.savefig('priceVsDistance.png')
plt.show()
#%%
def stories(row, colname): # colname can be 'rincome', 'income' etc
  thisstory = row[colname]
  if (thisstory < 20): return thisstory
  if (thisstory > 20): return np.nan
  return np.nan
# end function cleanDfIncome
print("\nReady to continue.")
# %%
FinalDC['STORIES'] = FinalDC.apply(stories, colname='STORIES', axis=1)
# %%
sns.regplot(x="STORIES", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"}, x_jitter=10, line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Stories")
plt.ylabel("Price")
plt.savefig('priceVsstories.png')
plt.show()
# %%
sns.regplot(x="metro25", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"}, x_jitter=10,line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Categorical: 1 is Near Metro")
plt.ylabel("Price")
plt.savefig('priceVsMetro50.png')
plt.show()
# %%
sns.regplot(x="metro50", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"},line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Categorical: 1 is Near Metro")
plt.ylabel("Price")
plt.savefig('priceVsMetro25.png')
plt.show()
# %%
sns.regplot(x="LANDAREA", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"},line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Amount of Land Area")
plt.ylabel("Price")
plt.savefig('priceVsLand.png')
plt.show()

# %%
sns.lmplot(x="LANDAREA", y="PRICE", hue = 'DistanceDummy', data=FinalDC, fit_reg = True, x_jitter=10)
plt.title("Price vs Bath Rooms")
plt.xlabel("Amount of Land Area")
plt.ylabel("Price")
plt.savefig('priceVsLand.png')
plt.show()
# %%
print(FinalDC.metro50.describe())
#%%
#####################################
####### STEPWISE SELECTION ##########
#####################################
FinalDC.columns
#%%


#%%
FinalDC['log_price'] = np.log2(FinalDC['PRICE'])
#%%
model4 = ols(formula='log_price ~ metro50 + STORIES + LANDAREA + CNDTN + BATHRM + HF_BATHRM + AC + ROOMS + NUM_UNITS + C(STRUCT) + EYB' , data=FinalDC)
model4Fit = model4.fit(cov_type='HC3')
print(model4Fit.summary())
#%%
formulaAll='log_price ~ metro50 + STORIES + LANDAREA + CNDTN + BATHRM + HF_BATHRM + AC + ROOMS + NUM_UNITS + C(STRUCT) + EYB' 
#%%
formulaA = 'log_price ~ metro50'
formulaB= 'log_price ~ STORIES'
formulaC= 'log_price ~ LANDAREA'
formulaD = 'log_price ~ CNDTN'
formulaE = 'log_price ~  BATHRM '
formulaF = 'log_price ~ HF_BATHRM'
formulaG= 'log_price ~ AC'
formulaH = 'log_price ~ROOMS'
formulaI= 'log_price ~NUM_UNITS'
formulaJ= 'log_price ~C(STRUCT)'
formulaK= 'log_price ~EYB'
formulaL = 'log_price ~roomsSQ'
formulaM = 'log_price ~bathSQ'
formulaN = 'log_price ~bedSQ'
#%%
modelA = ols(formula=formulaA, data=FinalDC)
modelAFit = modelA.fit(cov_type='HC3')

modelB = ols(formula=formulaB, data=FinalDC)
modelBFit = modelB.fit(cov_type='HC3')

modelC = ols(formula=formulaC, data=FinalDC)
modelCFit = modelC.fit(cov_type='HC3')

modelD = ols(formula=formulaD, data=FinalDC)
modelDFit = modelD.fit(cov_type='HC3')

modelE = ols(formula=formulaE, data=FinalDC)
modelEFit = modelE.fit(cov_type='HC3')

modelF = ols(formula=formulaF, data=FinalDC)
modelFFit = modelF.fit(cov_type='HC3')

modelG = ols(formula=formulaG, data=FinalDC)
modelGFit = modelG.fit(cov_type='HC3')

modelH = ols(formula=formulaH, data=FinalDC)
modelHFit = modelH.fit(cov_type='HC3')

modelI = ols(formula=formulaI, data=FinalDC)
modelIFit = modelI.fit(cov_type='HC3')

modelJ = ols(formula=formulaJ, data=FinalDC)
modelJFit = modelJ.fit(cov_type='HC3')

modelK = ols(formula=formulaK, data=FinalDC)
modelKFit = modelK.fit(cov_type='HC3')

modelL = ols(formula=formulaL, data=FinalDC)
modelLFit = modelL.fit(cov_type='HC3')

modelM = ols(formula=formulaM, data=FinalDC)
modelMFit = modelM.fit(cov_type='HC3')

modelN = ols(formula=formulaN, data=FinalDC)
modelNFit = modelN.fit(cov_type='HC3')
#%%

print('Model A:',modelAFit.rsquared)
print('Model B:',modelBFit.rsquared)
print('Model C:',modelCFit.rsquared)
print('Model D:',modelDFit.rsquared)
print('Model E:',modelEFit.rsquared)
print('Model F:',modelFFit.rsquared)
print('Model G' ,modelGFit.rsquared)
print('Model H:',modelHFit.rsquared)
print('Model I:',modelIFit.rsquared)
print('Model J:',modelJFit.rsquared)
print('Model K:',modelKFit.rsquared)
print('Model L:',modelKFit.rsquared)
print('Model M:',modelMFit.rsquared)
print('Model N:',modelNFit.rsquared)
# %%
max1= max(
modelAFit.rsquared,
modelBFit.rsquared,
modelCFit.rsquared,
modelDFit.rsquared,
modelEFit.rsquared,
modelFFit.rsquared,
modelGFit.rsquared,
modelHFit.rsquared,
modelIFit.rsquared,
modelJFit.rsquared,
modelKFit.rsquared,
modelLFit.rsquared,
modelMFit.rsquared,
modelNFit.rsquared)
print(max1)

#%%
formulaA = 'log_price ~ BATHRM+metro50'
formulaB= 'log_price ~ BATHRM+STORIES'
formulaC= 'log_price ~ BATHRM+LANDAREA'
formulaD = 'log_price ~ BATHRM+CNDTN'
# formulaE = 'log_price ~  BATHRM '
formulaF = 'log_price ~ BATHRM+HF_BATHRM'
formulaG= 'log_price ~BATHRM+ AC'
formulaH = 'log_price ~BATHRM+ROOMS'
formulaI= 'log_price ~BATHRM+NUM_UNITS'
formulaJ= 'log_price ~BATHRM+C(STRUCT)'
formulaK= 'log_price ~BATHRM+EYB'
formulaL = 'log_price ~BATHRM+roomsSQ'
formulaM = 'log_price ~BATHRM+bathSQ'
formulaN = 'log_price ~BATHRM+bedSQ'
# %%
modelA = ols(formula=formulaA, data=FinalDC)
modelAFit = modelA.fit(cov_type='HC3')

modelB = ols(formula=formulaB, data=FinalDC)
modelBFit = modelB.fit(cov_type='HC3')

modelC = ols(formula=formulaC, data=FinalDC)
modelCFit = modelC.fit(cov_type='HC3')

modelD = ols(formula=formulaD, data=FinalDC)
modelDFit = modelD.fit(cov_type='HC3')

# modelE = ols(formula=formulaE, data=FinalDC)
# modelEFit = modelE.fit(cov_type='HC3')

modelF = ols(formula=formulaF, data=FinalDC)
modelFFit = modelF.fit(cov_type='HC3')

modelG = ols(formula=formulaG, data=FinalDC)
modelGFit = modelG.fit(cov_type='HC3')

modelH = ols(formula=formulaH, data=FinalDC)
modelHFit = modelH.fit(cov_type='HC3')

modelI = ols(formula=formulaI, data=FinalDC)
modelIFit = modelI.fit(cov_type='HC3')

modelJ = ols(formula=formulaJ, data=FinalDC)
modelJFit = modelJ.fit(cov_type='HC3')

modelK = ols(formula=formulaK, data=FinalDC)
modelKFit = modelK.fit(cov_type='HC3')

modelL = ols(formula=formulaL, data=FinalDC)
modelLFit = modelL.fit(cov_type='HC3')

modelM = ols(formula=formulaM, data=FinalDC)
modelMFit = modelM.fit(cov_type='HC3')

modelN = ols(formula=formulaN, data=FinalDC)
modelNFit = modelN.fit(cov_type='HC3')
#%%
print('Model A:',modelAFit.rsquared)
print('Model B:',modelBFit.rsquared)
print('Model C:',modelCFit.rsquared)
print('Model D:',modelDFit.rsquared)
# print('Model E:',modelEFit.rsquared)
print('Model F:',modelFFit.rsquared)
print('Model G' ,modelGFit.rsquared)
print('Model H:',modelHFit.rsquared)
print('Model I:',modelIFit.rsquared)
print('Model J:',modelJFit.rsquared)
print('Model K:',modelKFit.rsquared)
print('Model L:',modelKFit.rsquared)
print('Model M:',modelMFit.rsquared)
print('Model N:',modelNFit.rsquared)
#%%
max2= max(modelAFit.rsquared,
modelBFit.rsquared,
modelCFit.rsquared,
modelDFit.rsquared,
# modelEFit.rsquared,
modelFFit.rsquared,
modelGFit.rsquared,
modelHFit.rsquared,
modelIFit.rsquared,
modelJFit.rsquared,
modelKFit.rsquared,
modelLFit.rsquared,
modelMFit.rsquared,
modelNFit.rsquared)
print(max2)
# %%
print( modelJFit.summary() )
# %%
formulaA = 'log_price ~ C(STRUCT)+BATHRM+metro50'
formulaB= 'log_price ~ C(STRUCT)+BATHRM+STORIES'
formulaC= 'log_price ~ C(STRUCT)+BATHRM+LANDAREA'
formulaD = 'log_price ~ C(STRUCT)+BATHRM+CNDTN'
# formulaE = 'log_price ~  BATHRM '
formulaF = 'log_price ~ C(STRUCT)+BATHRM+HF_BATHRM'
formulaG= 'log_price ~C(STRUCT)+BATHRM+ AC'
formulaH = 'log_price ~C(STRUCT)+BATHRM+ROOMS'
formulaI= 'log_price ~C(STRUCT)+BATHRM+NUM_UNITS'
# formulaJ= 'log_price ~BATHRM+C(STRUCT)'
formulaK= 'log_price ~C(STRUCT)+BATHRM+EYB'
formulaL = 'log_price ~C(STRUCT)+BATHRM+roomsSQ'
formulaM = 'log_price ~C(STRUCT)+ BATHRM+bathSQ'
formulaN = 'log_price ~C(STRUCT)+BATHRM+bedSQ'

#%%
modelA = ols(formula=formulaA, data=FinalDC)
modelAFit = modelA.fit(cov_type='HC3')

modelB = ols(formula=formulaB, data=FinalDC)
modelBFit = modelB.fit(cov_type='HC3')

modelC = ols(formula=formulaC, data=FinalDC)
modelCFit = modelC.fit(cov_type='HC3')

modelD = ols(formula=formulaD, data=FinalDC)
modelDFit = modelD.fit(cov_type='HC3')

# modelE = ols(formula=formulaE, data=FinalDC)
# modelEFit = modelE.fit(cov_type='HC3')

modelF = ols(formula=formulaF, data=FinalDC)
modelFFit = modelF.fit(cov_type='HC3')

modelG = ols(formula=formulaG, data=FinalDC)
modelGFit = modelG.fit(cov_type='HC3')

modelH = ols(formula=formulaH, data=FinalDC)
modelHFit = modelH.fit(cov_type='HC3')

modelI = ols(formula=formulaI, data=FinalDC)
modelIFit = modelI.fit(cov_type='HC3')

# modelJ = ols(formula=formulaJ, data=FinalDC)
# modelJFit = modelJ.fit(cov_type='HC3')

modelK = ols(formula=formulaK, data=FinalDC)
modelKFit = modelK.fit(cov_type='HC3')

modelL = ols(formula=formulaL, data=FinalDC)
modelLFit = modelL.fit(cov_type='HC3')

modelM = ols(formula=formulaM, data=FinalDC)
modelMFit = modelM.fit(cov_type='HC3')

modelN = ols(formula=formulaN, data=FinalDC)
modelNFit = modelN.fit(cov_type='HC3')
#%%
print('Model A:',modelAFit.rsquared)
print('Model B:',modelBFit.rsquared)
print('Model C:',modelCFit.rsquared)
print('Model D:',modelDFit.rsquared)
# print('Model E:',modelEFit.rsquared)
print('Model F:',modelFFit.rsquared)
print('Model G' ,modelGFit.rsquared)
print('Model H:',modelHFit.rsquared)
print('Model I:',modelIFit.rsquared)
# print('Model J:',modelJFit.rsquared)
print('Model K:',modelKFit.rsquared)
print('Model L:',modelKFit.rsquared)
print('Model M:',modelMFit.rsquared)
print('Model N:',modelNFit.rsquared)
#%%
max3= max(modelAFit.rsquared,
modelBFit.rsquared,
modelCFit.rsquared,
modelDFit.rsquared,
# modelEFit.rsquared,
modelFFit.rsquared,
modelGFit.rsquared,
modelHFit.rsquared,
modelIFit.rsquared,
# modelJFit.rsquared,
modelKFit.rsquared,
modelLFit.rsquared,
modelMFit.rsquared,
modelNFit.rsquared)
print(max3)
# %%
print( modelFFit.summary() )
#%%
#############
formulaA = 'log_price ~ HF_BATHRM+ C(STRUCT)+BATHRM+metro50'
formulaB= 'log_price ~  HF_BATHRM+ C(STRUCT)+BATHRM+STORIES'
formulaC= 'log_price ~  HF_BATHRM+ C(STRUCT)+BATHRM+LANDAREA'
formulaD = 'log_price ~  HF_BATHRM+ C(STRUCT)+BATHRM+CNDTN'
# formulaE = 'log_price ~  BATHRM '
# formulaF = 'log_price ~ C(STRUCT)+BATHRM+HF_BATHRM'
formulaG= 'log_price ~ HF_BATHRM+ C(STRUCT)+BATHRM+ AC'
formulaH = 'log_price ~ HF_BATHRM+ C(STRUCT)+BATHRM+ROOMS'
formulaI= 'log_price ~ HF_BATHRM+ C(STRUCT)+BATHRM+NUM_UNITS'
# formulaJ= 'log_price ~BATHRM+C(STRUCT)'
formulaK= 'log_price ~ HF_BATHRM+ C(STRUCT)+BATHRM+EYB'
formulaL = 'log_price ~ HF_BATHRM+ C(STRUCT)+BATHRM+roomsSQ'
formulaM = 'log_price ~ HF_BATHRM+ C(STRUCT)+ BATHRM+bathSQ'
formulaN = 'log_price ~ HF_BATHRM+ C(STRUCT)+BATHRM+bedSQ'
# %%
modelA = ols(formula=formulaA, data=FinalDC)
modelAFit = modelA.fit(cov_type='HC3')

modelB = ols(formula=formulaB, data=FinalDC)
modelBFit = modelB.fit(cov_type='HC3')

modelC = ols(formula=formulaC, data=FinalDC)
modelCFit = modelC.fit(cov_type='HC3')

modelD = ols(formula=formulaD, data=FinalDC)
modelDFit = modelD.fit(cov_type='HC3')

# modelE = ols(formula=formulaE, data=FinalDC)
# modelEFit = modelE.fit(cov_type='HC3')

# modelF = ols(formula=formulaF, data=FinalDC)
# modelFFit = modelF.fit(cov_type='HC3')

modelG = ols(formula=formulaG, data=FinalDC)
modelGFit = modelG.fit(cov_type='HC3')

modelH = ols(formula=formulaH, data=FinalDC)
modelHFit = modelH.fit(cov_type='HC3')

modelI = ols(formula=formulaI, data=FinalDC)
modelIFit = modelI.fit(cov_type='HC3')

# modelJ = ols(formula=formulaJ, data=FinalDC)
# modelJFit = modelJ.fit(cov_type='HC3')

modelK = ols(formula=formulaK, data=FinalDC)
modelKFit = modelK.fit(cov_type='HC3')

modelL = ols(formula=formulaL, data=FinalDC)
modelLFit = modelL.fit(cov_type='HC3')

modelM = ols(formula=formulaM, data=FinalDC)
modelMFit = modelM.fit(cov_type='HC3')

modelN = ols(formula=formulaN, data=FinalDC)
modelNFit = modelN.fit(cov_type='HC3')
# %%
print('Model A:',modelAFit.rsquared)
print('Model B:',modelBFit.rsquared)
print('Model C:',modelCFit.rsquared)
print('Model D:',modelDFit.rsquared)
# print('Model E:',modelEFit.rsquared)
# print('Model F:',modelFFit.rsquared)
print('Model G' ,modelGFit.rsquared)
print('Model H:',modelHFit.rsquared)
print('Model I:',modelIFit.rsquared)
# print('Model J:',modelJFit.rsquared)
print('Model K:',modelKFit.rsquared)
print('Model L:',modelKFit.rsquared)
print('Model M:',modelMFit.rsquared)
print('Model N:',modelNFit.rsquared)
#%%
max4= max(modelAFit.rsquared,
modelBFit.rsquared,
modelCFit.rsquared,
modelDFit.rsquared,
# modelEFit.rsquared,
# modelFFit.rsquared,
modelGFit.rsquared,
modelHFit.rsquared,
modelIFit.rsquared,
# modelJFit.rsquared,
modelKFit.rsquared,
modelLFit.rsquared,
modelMFit.rsquared,
modelNFit.rsquared)
print(max4)
# %%
print( modelHFit.summary() )
# %%
formulaA = 'log_price ~ ROOMS+ HF_BATHRM+ C(STRUCT)+BATHRM+metro50'
formulaB= 'log_price ~  ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+STORIES'
formulaC= 'log_price ~ ROOMS+ HF_BATHRM+ C(STRUCT)+BATHRM+LANDAREA'
formulaD = 'log_price ~  ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+CNDTN'
# formulaE = 'log_price ~  BATHRM '
# formulaF = 'log_price ~ C(STRUCT)+BATHRM+HF_BATHRM'
formulaG= 'log_price ~ ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+ AC'
# formulaH = 'log_price ~ HF_BATHRM+ C(STRUCT)+BATHRM+ROOMS'
formulaI= 'log_price ~ ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+NUM_UNITS'
# formulaJ= 'log_price ~BATHRM+C(STRUCT)'
formulaK= 'log_price ~ ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+EYB'
formulaL = 'log_price ~ ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+roomsSQ'
formulaM = 'log_price ~ ROOMS+HF_BATHRM+ C(STRUCT)+ BATHRM+bathSQ'
formulaN = 'log_price ~ ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+bedSQ'
# %%
modelA = ols(formula=formulaA, data=FinalDC)
modelAFit = modelA.fit(cov_type='HC3')

modelB = ols(formula=formulaB, data=FinalDC)
modelBFit = modelB.fit(cov_type='HC3')

modelC = ols(formula=formulaC, data=FinalDC)
modelCFit = modelC.fit(cov_type='HC3')

modelD = ols(formula=formulaD, data=FinalDC)
modelDFit = modelD.fit(cov_type='HC3')

# modelE = ols(formula=formulaE, data=FinalDC)
# modelEFit = modelE.fit(cov_type='HC3')

# modelF = ols(formula=formulaF, data=FinalDC)
# modelFFit = modelF.fit(cov_type='HC3')

modelG = ols(formula=formulaG, data=FinalDC)
modelGFit = modelG.fit(cov_type='HC3')

# modelH = ols(formula=formulaH, data=FinalDC)
# modelHFit = modelH.fit(cov_type='HC3')

modelI = ols(formula=formulaI, data=FinalDC)
modelIFit = modelI.fit(cov_type='HC3')

# modelJ = ols(formula=formulaJ, data=FinalDC)
# modelJFit = modelJ.fit(cov_type='HC3')

modelK = ols(formula=formulaK, data=FinalDC)
modelKFit = modelK.fit(cov_type='HC3')

modelL = ols(formula=formulaL, data=FinalDC)
modelLFit = modelL.fit(cov_type='HC3')

modelM = ols(formula=formulaM, data=FinalDC)
modelMFit = modelM.fit(cov_type='HC3')

modelN = ols(formula=formulaN, data=FinalDC)
modelNFit = modelN.fit(cov_type='HC3')


# %%
print('Model A:',modelAFit.rsquared)
print('Model B:',modelBFit.rsquared)
print('Model C:',modelCFit.rsquared)
print('Model D:',modelDFit.rsquared)
# print('Model E:',modelEFit.rsquared)
# print('Model F:',modelFFit.rsquared)
print('Model G' ,modelGFit.rsquared)
# print('Model H:',modelHFit.rsquared)
print('Model I:',modelIFit.rsquared)
# print('Model J:',modelJFit.rsquared)
print('Model K:',modelKFit.rsquared)
print('Model L:',modelKFit.rsquared)
print('Model M:',modelMFit.rsquared)
print('Model N:',modelNFit.rsquared)
#%%
max5= max(modelAFit.rsquared,
modelBFit.rsquared,
modelCFit.rsquared,
modelDFit.rsquared,
# modelEFit.rsquared,
# modelFFit.rsquared,
modelGFit.rsquared,
# modelHFit.rsquared,
modelIFit.rsquared,
# modelJFit.rsquared,
modelKFit.rsquared,
modelLFit.rsquared,
modelMFit.rsquared,
modelNFit.rsquared)
print(max5)
# %%
print( modelDFit.summary() )
# %%
formulaA = 'log_price ~ ROOMS+ HF_BATHRM+ C(STRUCT)+BATHRM+metro50'
formulaB= 'log_price ~  ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+STORIES'
formulaC= 'log_price ~ ROOMS+ HF_BATHRM+ C(STRUCT)+BATHRM+LANDAREA'
# formulaD = 'log_price ~  ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+CNDTN'
# formulaE = 'log_price ~  BATHRM '
# formulaF = 'log_price ~ C(STRUCT)+BATHRM+HF_BATHRM'
formulaG= 'log_price ~ ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+ AC'
# formulaH = 'log_price ~ HF_BATHRM+ C(STRUCT)+BATHRM+ROOMS'
formulaI= 'log_price ~ ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+NUM_UNITS'
# formulaJ= 'log_price ~BATHRM+C(STRUCT)'
formulaK= 'log_price ~ ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+EYB'
formulaL = 'log_price ~ ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+roomsSQ'
formulaM = 'log_price ~ ROOMS+HF_BATHRM+ C(STRUCT)+ BATHRM+bathSQ'
formulaN = 'log_price ~ ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+bedSQ'
# %%
modelA = ols(formula=formulaA, data=FinalDC)
modelAFit = modelA.fit(cov_type='HC3')

modelB = ols(formula=formulaB, data=FinalDC)
modelBFit = modelB.fit(cov_type='HC3')

modelC = ols(formula=formulaC, data=FinalDC)
modelCFit = modelC.fit(cov_type='HC3')

# modelD = ols(formula=formulaD, data=FinalDC)
# modelDFit = modelD.fit(cov_type='HC3')

# modelE = ols(formula=formulaE, data=FinalDC)
# modelEFit = modelE.fit(cov_type='HC3')

# modelF = ols(formula=formulaF, data=FinalDC)
# modelFFit = modelF.fit(cov_type='HC3')

modelG = ols(formula=formulaG, data=FinalDC)
modelGFit = modelG.fit(cov_type='HC3')

# modelH = ols(formula=formulaH, data=FinalDC)
# modelHFit = modelH.fit(cov_type='HC3')

modelI = ols(formula=formulaI, data=FinalDC)
modelIFit = modelI.fit(cov_type='HC3')

# modelJ = ols(formula=formulaJ, data=FinalDC)
# modelJFit = modelJ.fit(cov_type='HC3')

modelK = ols(formula=formulaK, data=FinalDC)
modelKFit = modelK.fit(cov_type='HC3')

modelL = ols(formula=formulaL, data=FinalDC)
modelLFit = modelL.fit(cov_type='HC3')

modelM = ols(formula=formulaM, data=FinalDC)
modelMFit = modelM.fit(cov_type='HC3')

modelN = ols(formula=formulaN, data=FinalDC)
modelNFit = modelN.fit(cov_type='HC3')
# %%
print('Model A:',modelAFit.rsquared)
print('Model B:',modelBFit.rsquared)
print('Model C:',modelCFit.rsquared)
# print('Model D:',modelDFit.rsquared)
# print('Model E:',modelEFit.rsquared)
# print('Model F:',modelFFit.rsquared)
print('Model G' ,modelGFit.rsquared)
# print('Model H:',modelHFit.rsquared)
print('Model I:',modelIFit.rsquared)
# print('Model J:',modelJFit.rsquared)
print('Model K:',modelKFit.rsquared)
print('Model L:',modelKFit.rsquared)
print('Model M:',modelMFit.rsquared)
print('Model N:',modelNFit.rsquared)
#%%
max6= max(modelAFit.rsquared,
modelBFit.rsquared,
modelCFit.rsquared,
# modelDFit.rsquared,
# modelEFit.rsquared,
# modelFFit.rsquared,
modelGFit.rsquared,
# modelHFit.rsquared,
modelIFit.rsquared,
# modelJFit.rsquared,
modelKFit.rsquared,
modelLFit.rsquared,
modelMFit.rsquared,
modelNFit.rsquared)
print(max6)
# %%
print( modelAFit.summary() )
# %%
# formulaA = 'log_price ~ ROOMS+ HF_BATHRM+ C(STRUCT)+BATHRM+metro50'
formulaB= 'log_price ~  metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+STORIES'
formulaC= 'log_price ~ metro50+ROOMS+ HF_BATHRM+ C(STRUCT)+BATHRM+LANDAREA'
# formulaD = 'log_price ~  ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+CNDTN'
# formulaE = 'log_price ~  BATHRM '
# formulaF = 'log_price ~ C(STRUCT)+BATHRM+HF_BATHRM'
formulaG= 'log_price ~ metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+ AC'
# formulaH = 'log_price ~ HF_BATHRM+ C(STRUCT)+BATHRM+ROOMS'
formulaI= 'log_price ~ metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+NUM_UNITS'
# formulaJ= 'log_price ~BATHRM+C(STRUCT)'
formulaK= 'log_price ~ metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+EYB'
formulaL = 'log_price ~ metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+roomsSQ'
formulaM = 'log_price ~ metro50+ROOMS+HF_BATHRM+ C(STRUCT)+ BATHRM+bathSQ'
formulaN = 'log_price ~ metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+bedSQ'
# %%

# modelA = ols(formula=formulaA, data=FinalDC)
# modelAFit = modelA.fit(cov_type='HC3')

modelB = ols(formula=formulaB, data=FinalDC)
modelBFit = modelB.fit(cov_type='HC3')

modelC = ols(formula=formulaC, data=FinalDC)
modelCFit = modelC.fit(cov_type='HC3')

# modelD = ols(formula=formulaD, data=FinalDC)
# modelDFit = modelD.fit(cov_type='HC3')

# modelE = ols(formula=formulaE, data=FinalDC)
# modelEFit = modelE.fit(cov_type='HC3')

# modelF = ols(formula=formulaF, data=FinalDC)
# modelFFit = modelF.fit(cov_type='HC3')

modelG = ols(formula=formulaG, data=FinalDC)
modelGFit = modelG.fit(cov_type='HC3')

# modelH = ols(formula=formulaH, data=FinalDC)
# modelHFit = modelH.fit(cov_type='HC3')

modelI = ols(formula=formulaI, data=FinalDC)
modelIFit = modelI.fit(cov_type='HC3')

# modelJ = ols(formula=formulaJ, data=FinalDC)
# modelJFit = modelJ.fit(cov_type='HC3')

modelK = ols(formula=formulaK, data=FinalDC)
modelKFit = modelK.fit(cov_type='HC3')

modelL = ols(formula=formulaL, data=FinalDC)
modelLFit = modelL.fit(cov_type='HC3')

modelM = ols(formula=formulaM, data=FinalDC)
modelMFit = modelM.fit(cov_type='HC3')

modelN = ols(formula=formulaN, data=FinalDC)
modelNFit = modelN.fit(cov_type='HC3')
# %%
# print('Model A:',modelAFit.rsquared)
print('Model B:',modelBFit.rsquared)
print('Model C:',modelCFit.rsquared)
# print('Model D:',modelDFit.rsquared)
# print('Model E:',modelEFit.rsquared)
# print('Model F:',modelFFit.rsquared)
print('Model G' ,modelGFit.rsquared)
# print('Model H:',modelHFit.rsquared)
print('Model I:',modelIFit.rsquared)
# print('Model J:',modelJFit.rsquared)
print('Model K:',modelKFit.rsquared)
print('Model L:',modelLFit.rsquared)
print('Model M:',modelMFit.rsquared)
print('Model N:',modelNFit.rsquared)
#%%
max7= max(
# modelAFit.rsquared,
modelBFit.rsquared,
modelCFit.rsquared,
# modelDFit.rsquared,
# modelEFit.rsquared,
# modelFFit.rsquared,
modelGFit.rsquared,
# modelHFit.rsquared,
modelIFit.rsquared,
# modelJFit.rsquared,
modelKFit.rsquared,
modelLFit.rsquared,
modelMFit.rsquared,
modelNFit.rsquared)
print(max7)
# %%
print( modelCFit.summary() )
#%%
# formulaA = 'log_price ~ ROOMS+ HF_BATHRM+ C(STRUCT)+BATHRM+metro50'
formulaB= 'log_price ~  LANDAREA+ metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+STORIES'
# formulaC= 'log_price ~ metro50+ROOMS+ HF_BATHRM+ C(STRUCT)+BATHRM+LANDAREA'
# formulaD = 'log_price ~  ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+CNDTN'
# formulaE = 'log_price ~  BATHRM '
# formulaF = 'log_price ~ C(STRUCT)+BATHRM+HF_BATHRM'
# formulaG= 'log_price ~ metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+ AC'
# formulaH = 'log_price ~ HF_BATHRM+ C(STRUCT)+BATHRM+ROOMS'
formulaI= 'log_price ~ LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+NUM_UNITS'
# formulaJ= 'log_price ~BATHRM+C(STRUCT)'
formulaK= 'log_price ~ LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+EYB'
formulaL = 'log_price ~ LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+roomsSQ'
formulaM = 'log_price ~ LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+ BATHRM+bathSQ'
formulaN = 'log_price ~ LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+bedSQ'
# %%

# modelA = ols(formula=formulaA, data=FinalDC)
# modelAFit = modelA.fit(cov_type='HC3')

modelB = ols(formula=formulaB, data=FinalDC)
modelBFit = modelB.fit(cov_type='HC3')

# modelC = ols(formula=formulaC, data=FinalDC)
# modelCFit = modelC.fit(cov_type='HC3')

# modelD = ols(formula=formulaD, data=FinalDC)
# modelDFit = modelD.fit(cov_type='HC3')

# modelE = ols(formula=formulaE, data=FinalDC)
# modelEFit = modelE.fit(cov_type='HC3')

# modelF = ols(formula=formulaF, data=FinalDC)
# modelFFit = modelF.fit(cov_type='HC3')

# modelG = ols(formula=formulaG, data=FinalDC)
# modelGFit = modelG.fit(cov_type='HC3')

# modelH = ols(formula=formulaH, data=FinalDC)
# modelHFit = modelH.fit(cov_type='HC3')

modelI = ols(formula=formulaI, data=FinalDC)
modelIFit = modelI.fit(cov_type='HC3')

# modelJ = ols(formula=formulaJ, data=FinalDC)
# modelJFit = modelJ.fit(cov_type='HC3')

modelK = ols(formula=formulaK, data=FinalDC)
modelKFit = modelK.fit(cov_type='HC3')

modelL = ols(formula=formulaL, data=FinalDC)
modelLFit = modelL.fit(cov_type='HC3')

modelM = ols(formula=formulaM, data=FinalDC)
modelMFit = modelM.fit(cov_type='HC3')

modelN = ols(formula=formulaN, data=FinalDC)
modelNFit = modelN.fit(cov_type='HC3')
#%%
# print('Model A:',modelAFit.rsquared)
print('Model B:',modelBFit.rsquared)
# print('Model C:',modelCFit.rsquared)
# print('Model D:',modelDFit.rsquared)
# print('Model E:',modelEFit.rsquared)
# print('Model F:',modelFFit.rsquared)
# print('Model G' ,modelGFit.rsquared)
# print('Model H:',modelHFit.rsquared)
print('Model I:',modelIFit.rsquared)
# print('Model J:',modelJFit.rsquared)
print('Model K:',modelKFit.rsquared)
print('Model L:',modelLFit.rsquared)
print('Model M:',modelMFit.rsquared)
print('Model N:',modelNFit.rsquared)
#%%
max8= max(
# modelAFit.rsquared,
modelBFit.rsquared,
# modelCFit.rsquared,
# modelDFit.rsquared,
# modelEFit.rsquared,
# modelFFit.rsquared,
# modelGFit.rsquared,
# modelHFit.rsquared,
modelIFit.rsquared,
# modelJFit.rsquared,
modelKFit.rsquared,
modelLFit.rsquared,
modelMFit.rsquared,
modelNFit.rsquared)
print(max8)
# %%
print( modelBFit.summary() )
# %%
# formulaB= 'log_price ~  LANDAREA+ metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+STORIES'
formulaI= 'log_price ~ STORIES+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+NUM_UNITS'
formulaK= 'log_price ~ STORIES+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+EYB'
formulaL = 'log_price ~ STORIES+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+roomsSQ'
formulaM = 'log_price ~ STORIES+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+ BATHRM+bathSQ'
formulaN = 'log_price ~ STORIES+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+bedSQ'
#%%

modelI = ols(formula=formulaI, data=FinalDC)
modelIFit = modelI.fit(cov_type='HC3')

modelK = ols(formula=formulaK, data=FinalDC)
modelKFit = modelK.fit(cov_type='HC3')

modelL = ols(formula=formulaL, data=FinalDC)
modelLFit = modelL.fit(cov_type='HC3')

modelM = ols(formula=formulaM, data=FinalDC)
modelMFit = modelM.fit(cov_type='HC3')

modelN = ols(formula=formulaN, data=FinalDC)
modelNFit = modelN.fit(cov_type='HC3')
#%%
# print('Model A:',modelAFit.rsquared)
# print('Model B:',modelBFit.rsquared)
# print('Model C:',modelCFit.rsquared)
# print('Model D:',modelDFit.rsquared)
# print('Model E:',modelEFit.rsquared)
# print('Model F:',modelFFit.rsquared)
# print('Model G' ,modelGFit.rsquared)
# print('Model H:',modelHFit.rsquared)
print('Model I:',modelIFit.rsquared)
# print('Model J:',modelJFit.rsquared)
print('Model K:',modelKFit.rsquared)
print('Model L:',modelLFit.rsquared)
print('Model M:',modelMFit.rsquared)
print('Model N:',modelNFit.rsquared)
#%%
max9= max(
# modelAFit.rsquared,
# modelBFit.rsquared,
# modelCFit.rsquared,
# modelDFit.rsquared,
# modelEFit.rsquared,
# modelFFit.rsquared,
# modelGFit.rsquared,
# modelHFit.rsquared,
modelIFit.rsquared,
# modelJFit.rsquared,
modelKFit.rsquared,
modelLFit.rsquared,
modelMFit.rsquared,
modelNFit.rsquared)
print(max9)
#%%
print( modelLFit.summary() )
# %%
formulaI= 'log_price ~ roomsSQ+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+NUM_UNITS'
formulaK= 'log_price ~ roomsSQ+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+EYB'
# formulaL = 'log_price ~ STORIES+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+roomsSQ'
formulaM = 'log_price ~ roomsSQ+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+ BATHRM+bathSQ'
formulaN = 'log_price ~ roomsSQ+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+bedSQ'
#%%
modelI = ols(formula=formulaI, data=FinalDC)
modelIFit = modelI.fit(cov_type='HC3')

# modelJ = ols(formula=formulaJ, data=FinalDC)
# modelJFit = modelJ.fit(cov_type='HC3')

modelK = ols(formula=formulaK, data=FinalDC)
modelKFit = modelK.fit(cov_type='HC3')

modelM = ols(formula=formulaM, data=FinalDC)
modelMFit = modelM.fit(cov_type='HC3')

modelN = ols(formula=formulaN, data=FinalDC)
modelNFit = modelN.fit(cov_type='HC3')

#%%
print('Model I:',modelIFit.rsquared)
# print('Model J:',modelJFit.rsquared)
print('Model K:',modelKFit.rsquared)
print('Model M:',modelMFit.rsquared)
print('Model N:',modelNFit.rsquared)
#%%
max10= max(

modelIFit.rsquared,
# modelJFit.rsquared,
modelKFit.rsquared,
# modelLFit.rsquared,
modelMFit.rsquared,
modelNFit.rsquared)
print(max10)
# %%
print( modelNFit.summary() )
# %%
formulaI= 'log_price ~ bedSQ+roomsSQ+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+NUM_UNITS'
formulaK= 'log_price ~ bedSQ+roomsSQ+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+EYB'
# formulaL = 'log_price ~ STORIES+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+roomsSQ'
formulaM = 'log_price ~ bedSQ+ roomsSQ+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+ BATHRM+bathSQ'
# formulaN = 'log_price ~ roomsSQ+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+bedSQ'
# %%
modelI = ols(formula=formulaI, data=FinalDC)
modelIFit = modelI.fit(cov_type='HC3')

modelK = ols(formula=formulaK, data=FinalDC)
modelKFit = modelK.fit(cov_type='HC3')

modelM = ols(formula=formulaM, data=FinalDC)
modelMFit = modelM.fit(cov_type='HC3')

# %%
print('Model I:',modelIFit.rsquared)
print('Model K:',modelKFit.rsquared)
print('Model M:',modelMFit.rsquared)
#%%
max11= max(
modelIFit.rsquared,
# modelJFit.rsquared,
modelKFit.rsquared,
# modelLFit.rsquared,
modelMFit.rsquared
)
print(max11)
# %%
print( modelIFit.summary() )
# %%
formulaI= 'log_price ~ bedSQ+roomsSQ+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM'
formulaK= 'log_price ~ bedSQ+roomsSQ+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+EYB'
# formulaL = 'log_price ~ STORIES+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+roomsSQ'
formulaM = 'log_price ~ bedSQ+ roomsSQ+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+ BATHRM+bathSQ'
# formulaN = 'log_price ~ roomsSQ+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+bedSQ'
#%%
modelI = ols(formula=formulaI, data=FinalDC)
modelIFit = modelI.fit(cov_type='HC3')

modelK = ols(formula=formulaK, data=FinalDC)
modelKFit = modelK.fit(cov_type='HC3')

modelM = ols(formula=formulaM, data=FinalDC)
modelMFit = modelM.fit(cov_type='HC3')
# %%
# formulaI= 'log_price ~ bedSQ+roomsSQ+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM'
formulaK= 'log_price ~ bedSQ+roomsSQ+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+EYB'
# formulaL = 'log_price ~ STORIES+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+roomsSQ'
formulaM = 'log_price ~ bedSQ+ roomsSQ+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+ BATHRM+bathSQ'
# formulaN = 'log_price ~ roomsSQ+LANDAREA+metro50+ROOMS+HF_BATHRM+ C(STRUCT)+BATHRM+bedSQ'
#%%
# modelI = ols(formula=formulaI, data=FinalDC)
# modelIFit = modelI.fit(cov_type='HC3')

modelK = ols(formula=formulaK, data=FinalDC)
modelKFit = modelK.fit(cov_type='HC3')

modelM = ols(formula=formulaM, data=FinalDC)
modelMFit = modelM.fit(cov_type='HC3')
#%%


print('Model M:',modelMFit.rsquared)
#%%
max11= max(
# modelIFit.rsquared,
# modelJFit.rsquared,
modelKFit.rsquared,
# modelLFit.rsquared,
modelMFit.rsquared
)
print(max11)
# %%
print( modelKFit.summary() )
# %%
#%%
####################################
######### EXTRA PLOTS ##############
####################################
#Regression Plot
sns.regplot(x="distance", y="newPrice", data=FinalDC, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.title("Price vs Distance")
plt.xlabel("Distance to the metro")
plt.ylabel("Price")
plt.savefig('DistanceRegplot.png')
plt.show()


#%%
#Scatter Plots
sns.scatterplot(x="distance", y="newPrice", data=FinalDC)
plt.show()
#%%
sns.scatterplot(x="DistanceDummy", y="newPrice", data=FinalDC)
plt.show()
#%%
#%%
#Regression Plot
sns.regplot(x="BEDRM", y="newPrice", data=FinalDC, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.title("Price vs Bed Rooms")
plt.xlabel("Number of Bed Rooms")
plt.ylabel("Price")
plt.savefig('BedRegplot.png')
plt.show()

#%%
#Scatter Plots
sns.scatterplot(x="BEDRM", y="newPrice", data=FinalDC)
plt.savefig('BedroomScatter.png')
plt.show()


sns.scatterplot(x="BEDRM", y="newPrice", data=FinalDC)
plt.show()

#%%

#Regression Plot
sns.regplot(x="BATHRM", y="newPrice", data=FinalDC, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.savefig('BathRegplot.png')
plt.show()
#%%
#Regression Plot
sns.regplot(x="HF_BATHRM", y="newPrice", data=FinalDC, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.savefig('RoomRegplot.png')
plt.show()