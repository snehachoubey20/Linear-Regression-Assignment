#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Importing Libraries


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')


# In[3]:


bike_df = pd.read_csv(r'C:\Users\sneha\Downloads\day (1).csv')
bike_df.head()


# In[5]:


bike_df.shape


# In[6]:


bike_df.describe()


# In[7]:


bike_df.info()


# In[8]:


bike_df.columns


# In[9]:


bike_df.dtypes


# In[10]:


bike_df.size


# In[11]:


bike_df.axes


# In[12]:


bike_df.ndim


# In[13]:


bike_df.values


# In[ ]:


###There are 730 rows and 16 columns in the data set. There are no null values in any of the columns.


# In[15]:


##days_old variable shows how old the business is.

bike_df['days_old'] = (pd.to_datetime(bike_df['dteday'],format= '%d-%m-%Y') - pd.to_datetime('01-01-2018',format= '%d-%m-%Y')).dt.days


# In[16]:


bike_df.head()


# In[ ]:


##drop columns not useful for Analysis:-
1)instant is just a row instance identifier.
2)dteday is removed as we have some of date features like mnth and year and weekday already in other columns and also for this analysis we will not consider day to day trend in demand for bikes.
3)casual and registered variables are not available at the time of prediction and also these describe the target variable cnt in a very trivial way target = casual + registered, which leads to data leakage.


# In[18]:


# Droping instant column as it is an index column which has nothing to do with target
bike_df.drop(['instant'], axis = 1, inplace = True)

# Dropping dteday as we already have month and weekday columns to work with
bike_df.drop(['dteday'], axis = 1, inplace = True)

# Dropping casual and registered columns as we have cnt column which is sum of the both that is the target column

bike_df.drop(['casual'], axis = 1, inplace = True)
bike_df.drop(['registered'], axis = 1, inplace = True)


# In[19]:


bike_df.head()


# In[20]:


bike_df.info()


# In[22]:


bike_df.season.value_counts()


# In[23]:


bike_df.weathersit.value_counts()


# In[24]:


bike_df.corr()


# In[ ]:


##we can see that features like season, mnth, weekday and weathersit are numbers although they should be non-numerical categories.


# In[25]:


##handling missing values:-
bike_df.isnull().sum()


# In[ ]:


## there are no missing values in the data frame. 


# In[ ]:


## Handling outliers


# In[27]:


bike_df.columns


# In[29]:


##finding unique values. 


bike_df.nunique()


# In[30]:


##ploting box plot for independent variables and continous values:-

cols = ['temp', 'atemp', 'hum', 'windspeed']
plt.figure(figsize=(20,6))

i = 1
for col in cols:
    plt.subplot(1,4,i)
    sns.boxplot(y=col, data=bike_df)
    i+=1


# In[ ]:


## no outliers found in the data set. 


# In[31]:


##EDA
###Convert season and weathersit to categorical types:-

bike_df.season.replace({1:"spring", 2:"summer", 3:"fall", 4:"winter"},inplace = True)

bike_df.weathersit.replace({1:'good',2:'moderate',3:'bad',4:'severe'},inplace = True)

bike_df.mnth = bike_df.mnth.replace({1: 'jan',2: 'feb',3: 'mar',4: 'apr',5: 'may',6: 'jun',
                  7: 'jul',8: 'aug',9: 'sept',10: 'oct',11: 'nov',12: 'dec'})

bike_df.weekday = bike_df.weekday.replace({0: 'sun',1: 'mon',2: 'tue',3: 'wed',4: 'thu',5: 'fri',6: 'sat'})
bike_df.head()


# In[33]:


##ploting pqair plots to see the linear relationship:-

plt.figure(figsize = (15,30))
sns.pairplot(data=bike_df,vars=['cnt', 'temp', 'atemp', 'hum','windspeed'])
plt.show()


# In[ ]:


##Inference:
##1)Looks like the temp and atemp has the highest corelation with the target variable cnt
##2)temp and atemp are highly co-related with each other


# In[35]:


##Visualising the Data to Find the Correlation between the Numerical Variable:-

plt.figure(figsize=(20,15))
sns.pairplot(bike_df)
plt.show()


# In[36]:


# Checking continuous variables relationship with each other
sns.heatmap(bike_df[['temp','atemp','hum','windspeed','cnt']].corr(), cmap='BuGn', annot = True)
plt.show()


# In[ ]:


##Here we see that temp and atemp has correlation more than .99 means almost 1 (highly correlated) and atemp seems to be derived from temp so atemp field can be dropped here only


# In[38]:


#Correlations for numeric variables
cor=bike_df.corr()
sns.heatmap(cor, cmap="YlGnBu", annot = True)
plt.show()


# In[39]:


#Calculate Correlation
corr = bike_df.corr()
plt.figure(figsize=(25,10))

#Draw Heatmap of correlation
sns.heatmap(corr,annot=True, cmap='YlGnBu' )
plt.show()


# In[ ]:


##From the correlation map, temp, atemp and days_old seems to be highly correlated and only should variable can be considered for the model. However let us elminate it based on the Variance Inflation Factor later during the model building.
##We also see Target variable has a linear relationship with some of the indeptendent variables. Good sign for building a linear regression Model.


# In[41]:


vars_cat = ['season','yr','mnth','holiday','weekday','workingday','weathersit']
plt.figure(figsize=(15, 15))
for i in enumerate(vars_cat):
    plt.subplot(3,3,i[0]+1)
    sns.boxplot(data=bike_df, x=i[1], y='cnt')
plt.show()


# In[ ]:


####Here many insights can be drawn from the plots

##1. Season: 3:fall has highest demand for rental bikes
##2. I see that demand for next year has grown
##3. Demand is continuously growing each month till June. September month has highest demand. After September, demand is        decreasing
##4. When there is a holiday, demand has decreased.
##5. Weekday is not giving clear picture abount demand.
##6. The clear weathershit has highest demand
##7. During September, bike sharing is more. During the year end and beginning, it is less, could be due to extereme            weather conditions


# In[43]:


plt.figure(figsize=(6,5),dpi=110)
plt.title("Cnt vs Temp",fontsize=16)
sns.regplot(data=bike_df,y="cnt",x="temp")
plt.xlabel("Temperature")
plt.show()


# In[ ]:


##Inference:
##1)Demand for bikes is positively correlated to temp.
##2)We can see that cnt is linearly increasing with temp indicating linear relation.


# In[45]:


plt.figure(figsize=(6,5),dpi=110)
plt.title("Cnt vs Hum",fontsize=16)
sns.regplot(data=bike_df,y="cnt",x="hum")
plt.xlabel("Humidity")
plt.show()


# In[ ]:


##Inference:
##1)Hum is values are more scattered around.
##2)Although we can see cnt decreasing with increase in humidity.


# In[46]:


plt.figure(figsize=(6,5),dpi=110)
plt.title("Cnt vs Windspeed",fontsize=16)
sns.regplot(data=bike_df,y="cnt",x="windspeed")
plt.show()


# In[ ]:


##Inference:
##1)Windspeed is values are more scattered around.
##2)Although we can see cnt decreasing with increase in windspeed.


# In[47]:


num_features = ["temp","atemp","hum","windspeed","cnt"]
plt.figure(figsize=(15,8),dpi=130)
plt.title("Correlation of numeric features",fontsize=16)
sns.heatmap(bike_df[num_features].corr(),annot= True,cmap="mako")
plt.show()


# In[ ]:


##Inference:
##1)Temp and Atemp are highly correlated, we can take an action to remove one of them, but lets keep them for further analysis.
##2)Temp and Atemp also have high correlation with cnt variable.


# In[48]:


bike_df.describe()


# In[ ]:


##Data Preparation for Linear Regression


##Create dummy variables for all categorical variables


# In[50]:


bike_df = pd.get_dummies(data=bike_df,columns=["season","mnth","weekday"],drop_first=True)
bike_df = pd.get_dummies(data=bike_df,columns=["weathersit"])


# In[52]:


bike_df.columns


# In[53]:


bike_df.head()


# In[ ]:


## Model Building



##Split Data into training and test


# In[55]:


bike_df.shape


# In[57]:


#y to contain only target variable
y=bike_df.pop('cnt')

#X is all remainign variable also our independent variables
X=bike_df

#Train Test split with 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[58]:


#Inspect independent variables
X.head()


# In[59]:


# Checking shape and size for train and test
print(X_train.shape)
print(X_test.shape)


# In[ ]:


##To make all features in same scale to interpret easily

##Following columns are continous to be scaled temp,hum,windspeed


# In[60]:


# Importing required library
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler


# In[61]:


# Let us scale continuous variables
num_vars = ['temp','atemp','hum','windspeed','days_old']

#Use Normalized scaler to scale
scaler = MinMaxScaler()

#Fit and transform training set only
X_train[num_vars] = scaler.fit_transform(X_train[num_vars])


# In[62]:


#Inspect stats fro Training set after scaling
X_train.describe()


# In[63]:


X_train.head()


# In[ ]:


## Build a Model using RFE and Automated approach
#3Use RFE to eliminate some columns


# In[64]:


# Build a Lienar Regression model using SKLearn for RFE
lr = LinearRegression()
lr.fit(X_train,y_train)


# In[65]:


#Cut down number of features to 15 using automated approach
rfe = RFE(lr,15)
rfe.fit(X_train,y_train)


# In[66]:


#Columns selected by RFE and their weights
list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[ ]:


## Manual elimination
## Function to build a model using statsmodel api


# In[67]:


#Function to build a model using statsmodel api - Takes the columns to be selected for model as a parameter
def build_model(cols):
    X_train_sm = sm.add_constant(X_train[cols])
    lm = sm.OLS(y_train, X_train_sm).fit()
    print(lm.summary())
    return lm


# In[68]:


#Function to calculate VIFs and print them -Takes the columns for which VIF to be calcualted as a parameter
def get_vif(cols):
    df1 = X_train[cols]
    vif = pd.DataFrame()
    vif['Features'] = df1.columns
    vif['VIF'] = [variance_inflation_factor(df1.values, i) for i in range(df1.shape[1])]
    vif['VIF'] = round(vif['VIF'],2)
    print(vif.sort_values(by='VIF',ascending=False))


# In[69]:


#Print Columns selected by RFE. We will start with these columns for manual elimination
X_train.columns[rfe.support_]


# In[70]:


# Features not selected by RFE
X_train.columns[~rfe.support_]


# In[71]:


# Taking 15 columns supported by RFE for regression
X_train_rfe = X_train[['yr', 'holiday', 'workingday', 'temp', 'hum', 'windspeed', 'season_spring',
       'season_summer', 'season_winter', 'mnth_jan', 'mnth_jul', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']]


# In[72]:


X_train_rfe.shape


# In[ ]:


## Build Model
## Model 1 - Start with all variables selected by RFE


# In[73]:


#Selected columns for Model 1 - all columns selected by RFE
cols = ['yr', 'holiday', 'workingday', 'temp', 'hum', 'windspeed', 'season_spring',
       'season_summer', 'season_winter', 'mnth_jan', 'mnth_jul', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']

build_model(cols)
get_vif(cols)


# In[75]:


# Checking correlation of features selected by RFE with target column. 
# Also to check impact of different features on target.
plt.figure(figsize = (15,10))
sns.heatmap(bike_df[['yr', 'holiday', 'workingday', 'temp', 'hum', 'windspeed', 'season_spring',
       'season_summer', 'season_winter', 'mnth_jan', 'mnth_jul', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']].corr(), cmap='GnBu', annot=True)
plt.show()


# In[76]:


# Dropping the variable mnth_jan as it has negative coefficient and is insignificant as it has high p-value
cols = ['yr', 'holiday', 'workingday', 'temp', 'hum', 'windspeed', 'season_spring',
       'season_summer', 'season_winter', 'mnth_jul', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']
build_model(cols)
get_vif(cols)


# In[77]:


# Dropping the variable hum as it has negative coefficient and is insignificant as it has high p-value
cols = ['yr', 'holiday', 'workingday', 'temp', 'windspeed', 'season_spring',
       'season_summer', 'season_winter', 'mnth_jul', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']
build_model(cols)
get_vif(cols)


# In[78]:


# Dropping the variable holiday as it has negative coefficient and is insignificant as it has high p-value
cols = ['yr', 'workingday', 'temp', 'windspeed', 'season_spring',
       'season_summer', 'season_winter', 'mnth_jul', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']
build_model(cols)
get_vif(cols)


# In[79]:


# Dropping the variable mnth_jul as it has negative coefficient and is insignificant as it has high p-value
cols = ['yr', 'workingday', 'temp', 'windspeed', 'season_spring',
       'season_summer', 'season_winter', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']
build_model(cols)
get_vif(cols)


# In[80]:


# Dropping the variable temp as it has negative coefficient and is insignificant as it has high p-value
cols = ['yr', 'workingday', 'windspeed', 'season_spring',
       'season_summer', 'season_winter', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']
build_model(cols)
get_vif(cols)


# In[81]:


## Trying to replace July with spring as both were highly correlated

cols = ['yr', 'workingday', 'windspeed', 'mnth_jul',
       'season_summer', 'season_winter', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']
build_model(cols)
get_vif(cols)


# In[82]:


## Trying to replace July with spring as both were highly correlated

cols = ['yr', 'workingday', 'windspeed', 'mnth_jul',
       'season_summer', 'season_winter', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']
build_model(cols)
get_vif(cols)


# In[83]:


# Removing windspeed with spring as windspeed was highly correlated with temp
cols = ['yr', 'workingday', 'season_spring', 'mnth_jul',
       'season_summer', 'season_winter', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']
build_model(cols)
get_vif(cols)


# In[84]:


# using the weekend "Sunday" which was dropped during RFE instead of Saturday.

cols = ['yr', 'workingday', 'season_spring', 'mnth_jul',
       'season_summer', 'season_winter', 'mnth_sept', 'weekday_sun',
       'weathersit_bad', 'weathersit_moderate']
build_model(cols)
get_vif(cols)


# In[85]:


# adding temp and removed 'season_summer' and 'workingday'
cols = ['yr', 'season_spring', 'mnth_jul',
        'season_winter', 'mnth_sept', 'weekday_sun',
       'weathersit_bad', 'weathersit_moderate', 'temp']

#['yr', 'holiday','temp', 'spring','winter', 'July','September','Sunday','Light_Snow_Rain','Mist_Clody']
build_model(cols)
get_vif(cols)


# In[ ]:


##Inference_
###Here VIF seems to be almost accepted. p-value for all the features is almost 0.0 and R2 is 0.821 Let us select Model 11 as our final as it has all important statistics high (R-square, Adjusted R-squared and F-statistic), along with no insignificant variables and no multi coliinear (high VIF) variables. Difference between R-squared and Adjusted R-squared values for this model is veryless, which also means that there are no additional parameters that can be removed from this model.


# In[86]:


#Build a model with all columns to select features automatically
def build_model_sk(X,y):
    lr1 = LinearRegression()
    lr1.fit(X,y)
    return lr1


# In[87]:


#Let us build the finalmodel using sklearn
cols = ['yr', 'season_spring', 'mnth_jul',
        'season_winter', 'mnth_sept', 'weekday_sun',
       'weathersit_bad', 'weathersit_moderate', 'temp']

#Build a model with above columns
lr = build_model_sk(X_train[cols],y_train)
print(lr.intercept_,lr.coef_)


# In[ ]:


##Model Evaluation


##Residucal Analysis


# In[88]:


y_train_pred = lr.predict(X_train[cols])


# In[89]:


#Plot a histogram of the error terms
def plot_res_dist(act, pred):
    sns.distplot(act-pred)
    plt.title('Error Terms')
    plt.xlabel('Errors')


# In[90]:


plot_res_dist(y_train, y_train_pred)


# In[ ]:


##Errors are normally distribured here with mean 0. So everything seems to be fine


# In[91]:


##Actual vs Predicted
c = [i for i in range(0,len(X_train),1)]
plt.plot(c,y_train, color="blue")
plt.plot(c,y_train_pred, color="red")
plt.suptitle('Actual vs Predicted', fontsize = 15)
plt.xlabel('Index')
plt.ylabel('Demands')
plt.show()


# In[ ]:


##Actual and Predicted result following almost the same pattern so this model seems ok


# In[92]:


# Error Terms
c = [i for i in range(0,len(X_train),1)]
plt.plot(c,y_train-y_train_pred)
plt.suptitle('Error Terms', fontsize = 15)
plt.xlabel('Index')
plt.ylabel('y_train-y_train_pred')
plt.show()


# In[93]:


##Here,If we see the error terms are independent of each other.


#Print R-squared Value
r2_score(y_train,y_train_pred)


# In[ ]:


##Inference:-
##R2 Same as we obtained for our final model


# In[94]:


# scatter plot for the leniarity check
residual = (y_train - y_train_pred)
plt.scatter(y_train,residual)
plt.ylabel("y_train")
plt.xlabel("Residual")
plt.show()


# In[95]:


#Scale variables in X_test
num_vars = ['temp','atemp','hum','windspeed','days_old']

#Test data to be transformed only, no fitting
X_test[num_vars] = scaler.transform(X_test[num_vars])


# In[96]:


#Columns from our final model
cols = ['yr', 'season_spring', 'mnth_jul',
        'season_winter', 'mnth_sept', 'weekday_sun',
       'weathersit_bad', 'weathersit_moderate', 'temp']

#Predict the values for test data
y_test_pred = lr.predict(X_test[cols])


# In[97]:


# Find out the R squared value between test and predicted test data sets.  
r2_score(y_test,y_test_pred)


# In[ ]:


##Inference
#3R2 value for predictions on test data (0.815) is almost same as R2 value of train data(0.818). This is a good R-squared value, hence we can see our model is performing good even on unseen data (test data)


# In[98]:


# Plotting y_test and y_test_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, y_test_pred)
fig.suptitle('y_test vs y_test_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y_test', fontsize = 18)                          # X-label
plt.ylabel('y_test_pred', fontsize = 16)


# In[ ]:


##Inference
##We can observe that variance of the residuals (error terms) is constant across predictions. i.e error term does not vary much as the value of the predictor variable changes.


# In[99]:


#Function to plot Actual vs Predicted
#Takes Actual and PRedicted values as input along with the scale and Title to indicate which data
def plot_act_pred(act,pred,scale,dataname):
    c = [i for i in range(1,scale,1)]
    fig = plt.figure(figsize=(14,5))
    plt.plot(c,act, color="blue", linewidth=2.5, linestyle="-")
    plt.plot(c,pred, color="red",  linewidth=2.5, linestyle="-")
    fig.suptitle('Actual and Predicted - '+dataname, fontsize=20)              # Plot heading 
    plt.xlabel('Index', fontsize=18)                               # X-label
    plt.ylabel('Counts', fontsize=16)                               # Y-label


# In[100]:


#Plot Actual vs Predicted for Test Data
plot_act_pred(y_test,y_test_pred,len(y_test)+1,'Test Data')


# In[ ]:


##Inference
##As we can see predictions for test data is very close to actuals


# In[101]:


# Error terms
def plot_err_terms(act,pred):
    c = [i for i in range(1,220,1)]
    fig = plt.figure(figsize=(14,5))
    plt.plot(c,act-pred, color="blue", marker='o', linewidth=2.5, linestyle="")
    fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
    plt.xlabel('Index', fontsize=18)                      # X-label
    plt.ylabel('Counts - Predicted Counts', fontsize=16)                # Y-label


# In[102]:


#Plot error terms for test data
plot_err_terms(y_test,y_test_pred)


# In[ ]:


##Inference
##As we can see the error terms are randomly distributed and there is no pattern which means the output is explained well by the model and there are no other parameters that can explain the model better.


# In[103]:


# Checking data before scaling
bike_df.head()


# In[ ]:


##Intrepretting the Model
##Let us go with interpretting the RFE with Manual model results as we give more importance to imputation


# In[104]:


#Let us rebuild the final model of manual + rfe approach using statsmodel to interpret it
cols = ['yr', 'season_spring', 'mnth_jul',
        'season_winter', 'mnth_sept', 'weekday_sun',
       'weathersit_bad', 'weathersit_moderate', 'temp']

lm = build_model(cols)


# In[ ]:


##Interepretation of results
##Analysing the above model, the comapany should focus on the following features:
##1)Company should focus on expanding business during Spring.
##2)Company should focus on expanding business during September.
##3)Based on previous data it is expected to have a boom in number of users once situation comes back to normal, compared to 2019.
##4)There would be less bookings during Light Snow or Rain, they could probably use this time to serive the bikes without having business impact.
##5)Hence when the situation comes back to normal, the company should come up with new offers during spring when the weather is pleasant and also advertise a little for September as this is when business would be at its best.

##Conclusion
##Significant variables to predict the demand for shared bikes

holiday
temp
hum
windspeed
Season
months(January, July, September, November, December)
Year (2019)
Sunday
weathersit( Light Snow, Mist + Cloudy)

