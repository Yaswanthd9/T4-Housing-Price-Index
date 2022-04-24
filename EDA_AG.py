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

# Creating a dummy column for EDA 
def DistanceDummy(distance): # colname can be 'rincome', 'income' etc
  
  if distance <= 0.50: return 1
  if distance > 0.50 and distance <= 1: return 2
  if distance > 1: return 3
  else: return 'NA'

#Creating the new column
FinalDC['DistanceDummy'] = FinalDC['distance'].apply(DistanceDummy)

#%%
# def PRICE(row, colname): # colname can be 'rincome', 'income' etc
#   thisprice = row[colname]
#   if thisprice == 1: return np.nan
#   if thisprice > 1: return thisprice
#   else: return np.nan

# FinalDC.dropna(inplace=True)


# #Creating the new column
# FinalDC['DistanceDummy'] = FinalDC['distance'].apply(DistanceDummy)


#%%
# display the dataframe
print(FinalDC)
  
#%%

#PLOTS FOR DISTANCE
# Overall observation: Price decreased as distance increased
#%%
#Violin Plot 
sns.violinplot(x="DistanceDummy", y="PRICE", data= FinalDC, scale="width")
plt.title("Price vs Distance")
plt.xticks(range(3), ['0.25', '0.50', 'Beyond 0.50'])
plt.xlabel("Distance to the metro")
plt.ylabel("Price")

plt.show()

#%%

# Joint Plot
sns.jointplot(x="distance", y="PRICE", data=FinalDC, color = 'blue', kind='reg', line_kws={'color':'green'})
plt.title("Price vs Distance")
plt.xlabel("Distance to the metro")
plt.ylabel("Price")
x= ['0.5', '1', 'Greater than 1']
default_x_ticks = range(len(x))
plt.xticks(default_x_ticks, x)
plt.savefig('DistanceJoint.png')
plt.show()


#%%
#Regression Plot
sns.regplot(x="distance", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.title("Price vs Distance")
plt.xlabel("Distance to the metro")
plt.ylabel("Price")
plt.savefig('DistanceRegplot.png')
plt.show()


#%%
#Scatter Plots
sns.scatterplot(x="distance", y="PRICE", data=FinalDC)
plt.show()
#%%
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
plt.savefig('BedViolin.png')
plt.show()

#%%

# Joint Plot
sns.jointplot(x="BEDRM", y="PRICE", data=FinalDC, color = 'blue', kind='reg', line_kws={'color':'green'})
plt.title("Price vs Bed Rooms")
plt.xlabel("Number of Bed Rooms")
plt.ylabel("Price")
plt.savefig('BedJoint.png')
plt.show()

#%%
#Regression Plot
sns.regplot(x="BEDRM", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.title("Price vs Bed Rooms")
plt.xlabel("Number of Bed Rooms")
plt.ylabel("Price")
plt.savefig('BedRegplot.png')
plt.show()

#%%
#Scatter Plots
sns.scatterplot(x="BEDRM", y="PRICE", data=FinalDC)
plt.savefig('BedroomScatter.png')
plt.show()


sns.scatterplot(x="BEDRM", y="PRICE", data=FinalDC)
plt.show()


# %%

#PLOTS FOR BATHROOMS
# Overall observation: price increased as the number of bedrooms increased

#Violin Plot 
sns.violinplot(x="BATHRM", y="PRICE", data= FinalDC, scale="width")
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.savefig('BathViolin.png')
plt.show()

#%%

# Joint Plot
sns.jointplot(x="BATHRM", y="PRICE", data=FinalDC, color = 'blue', kind='reg', line_kws={'color':'green'})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.savefig('BathJoint.png')
plt.show()
#%%

#Regression Plot
sns.regplot(x="BATHRM", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.savefig('BathRegplot.png')
plt.show()


# %%

#PLOTS FOR Half BATHROOMS
# Overall observation: price increased as the number of half baths increased
#%%
#Violin Plot 
sns.violinplot(x="HF_BATHRM", y="PRICE", data= FinalDC, scale="width")
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.savefig('halfViolin.png')
plt.show()


#%%
# Joint Plot
sns.jointplot(x="HF_BATHRM", y="PRICE", data=FinalDC, color = 'blue', kind='reg', line_kws={'color':'green'})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.savefig('hfJoint.png')
plt.show()

#%%
#Regression Plot
sns.regplot(x="HF_BATHRM", y="PRICE", data=FinalDC, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.savefig('RoomRegplot.png')
plt.show()


# %%
#PLOTS FOR STORIES

FinalDC['STORIES'] = FinalDC['STORIES'].round()
FinalDC['STORIES'] = FinalDC['STORIES'].dropna()
print(FinalDC.STORIES)
#%%
#Violin Plot 
sns.violinplot(x="STORIES", y="PRICE", data= FinalDC, scale="width")
plt.title("Price vs Stories")
plt.xlabel("Number of Stories")
plt.ylabel("Price")
plt.savefig('StoriesViolin.png')
plt.show()

# %%
#PLOTS FOR AC

#Violin Plot 
sns.violinplot(x="AC", y="PRICE", data= FinalDC, scale="width")
plt.title("Price vs Bath Rooms")
plt.xlabel("Number of Bath Rooms")
plt.ylabel("Price")
plt.savefig('BathViolin.png')
plt.show()


# %%

#PLOTS FOR Condition

#Violin Plot 
sns.violinplot(x="CNDTN", y="PRICE", data= FinalDC, scale="width")
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
# stargazer = Stargazer([glmmodel1Fit])
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
model7 = ols(formula='log_price ~ metro50 + ROOMS+ roomsSQ + BATHRM + bathSQ  + BEDRM + bedSQ + NUM_UNITS +C(STRUCT) + CNDTN' , data=FinalDC)
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
y= FinalDC["PRICE"]
x= FinalDC.drop(['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'CITY', 'STATE', 'ZIPCODE', 'NATIONALGRID','LATITUDE','LONGITUDE', 'SQUARE', 'X', 'Y', 'QUADRANT', 'year', 'distance2',  '.25metro', 'PRICE' ] , axis=1)

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
# %%
max1= max(modelAFit.rsquared,
modelBFit.rsquared,
modelCFit.rsquared,
modelDFit.rsquared,
modelEFit.rsquared,
modelFFit.rsquared,
modelGFit.rsquared,
modelHFit.rsquared,
modelIFit.rsquared,
modelJFit.rsquared,
modelKFit.rsquared)
print(max1)

#%%
formulaA = 'log_price ~ BATHRM +metro50'
formulaB= 'log_price ~ BATHRM +STORIES'
formulaC= 'log_price ~ BATHRM +LANDAREA'
formulaD = 'log_price ~ BATHRM +CNDTN'
# formulaE = 'log_price ~  BATHRM '
formulaF = 'log_price ~ BATHRM +HF_BATHRM'
formulaG= 'log_price ~BATHRM + AC'
formulaH = 'log_price ~BATHRM +ROOMS'
formulaI= 'log_price ~BATHRM +NUM_UNITS'
formulaJ= 'log_price ~BATHRM +C(STRUCT)'
formulaK= 'log_price ~BATHRM +EYB'
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
modelKFit.rsquared)
print(max2)
# %%
print( modelJFit.summary() )
# %%
formulaA = 'log_price ~ C(STRUCT) +BATHRM +metro50'
formulaB= 'log_price ~ C(STRUCT) +BATHRM +STORIES'
formulaC= 'log_price ~ C(STRUCT) +BATHRM +LANDAREA'
formulaD = 'log_price ~ C(STRUCT) +BATHRM +CNDTN'
# formulaE = 'log_price ~  BATHRM '
formulaF = 'log_price ~ C(STRUCT) +BATHRM +HF_BATHRM'
formulaG= 'log_price ~C(STRUCT) +BATHRM + AC'
formulaH = 'log_price ~C(STRUCT) +BATHRM +ROOMS'
formulaI= 'log_price ~C(STRUCT) +BATHRM +NUM_UNITS'
# formulaJ= 'log_price ~BATHRM +C(STRUCT)'
formulaK= 'log_price ~C(STRUCT) +BATHRM +EYB'

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
modelKFit.rsquared)
print(max3)
# %%
print( modelFFit.summary() )
#%%
#############
formulaA = 'log_price ~ HF_BATHRM+ C(STRUCT) +BATHRM +metro50'
formulaB= 'log_price ~  HF_BATHRM+C(STRUCT) +BATHRM +STORIES'
formulaC= 'log_price ~  HF_BATHRM+C(STRUCT) +BATHRM +LANDAREA'
formulaD = 'log_price ~  HF_BATHRM+ C(STRUCT) +BATHRM +CNDTN'
# formulaE = 'log_price ~  BATHRM '
# formulaF = 'log_price ~ C(STRUCT) +BATHRM +HF_BATHRM'
formulaG= 'log_price ~  HF_BATHRM+C(STRUCT) +BATHRM + AC'
formulaH = 'log_price ~  HF_BATHRM+ C(STRUCT) +BATHRM +ROOMS'
formulaI= 'log_price ~  HF_BATHRM+C(STRUCT) +BATHRM +NUM_UNITS'
# formulaJ= 'log_price ~BATHRM +C(STRUCT)'
formulaK= 'log_price ~  HF_BATHRM+ C(STRUCT) +BATHRM +EYB'
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
modelKFit.rsquared)
print(max4)
# %%
print( modelHFit.summary() )
# %%
