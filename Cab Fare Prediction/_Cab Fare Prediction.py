#!/usr/bin/env python
# coding: utf-8

# In[1]:


#working directory
import os 
os.chdir("D:/edvisor data science path/Project_2_python")
os.getcwd()


# In[2]:


#import the usefull libraries 
import pandas as pd
import matplotlib as mlt
import numpy as np
from fancyimpute import KNN
import matplotlib.pyplot as plt
import seaborn as sns
from random import randrange,uniform
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier 
import statsmodels.api as sm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import scipy.stats 
from scipy.stats import chi2_contingency
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.metrics import accuracy_score


# In[3]:


#for visualizations(to use ggplot) import some usefull libraries 
from pandas.api.types import CategoricalDtype
from plotnine import *
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


training_data=pd.read_csv("train_cab.csv",sep=',')


# In[5]:


test_data=pd.read_csv("test.csv",sep=',')


# In[5]:


training_data.shape


# In[7]:


training_data.index


# In[8]:


training_data.columns


# In[6]:


training_data.isnull().sum()


# In[10]:


training_data['fare_amount'].describe()


# In[11]:


training_data['pickup_datetime'].describe()


# In[12]:


training_data['pickup_longitude'].describe()
    


# In[13]:


training_data['pickup_latitude'].describe()


# In[14]:


training_data['dropoff_longitude'].describe()


# In[15]:


training_data['dropoff_latitude'].describe()


# In[16]:


training_data['passenger_count'].describe()


# In[6]:


#exploratory data analysis
#changing the variable datatypes into desired datatypes we need the target variable into numeric formater and the datetime varable into datetime formate and we need to ananlyze the lattitude and longitude variable as well 

training_data['fare_amount'] = pd.to_numeric(training_data['fare_amount'], errors='coerce')


# In[7]:


training_data['fare_amount'].describe()


# In[7]:


import datetime 


# In[8]:


training_data['pickup_datetime'] = pd.to_datetime(training_data['pickup_datetime'],errors='coerce')


# In[9]:


training_data['pickup_datetime']=pd.to_numeric(training_data['pickup_datetime'],errors='coerce')


# In[10]:


training_data['pickup_datetime'].describe()


# In[11]:


training_data


# In[36]:


training_data['pickup_datetime'].describe()


# In[14]:


training_data.info()


# In[13]:


#Missing value analysis
#create dataframe with missing percentage

missing_val=pd.DataFrame(training_data.isnull().sum())


# In[17]:


#restting the index
missing_val=missing_val.reset_index()


# In[19]:


#rename the variables 
missing_val=missing_val.rename(columns={'index':'variables',0:'missing_percentage'})


# In[20]:


missing_val


# In[21]:


#calculate percentage
missing_val['missing_percentage']=(missing_val['missing_percentage']/len(training_data))*100


# In[22]:


missing_val


# In[23]:


#sort in descending order
missing_val=missing_val.sort_values('missing_percentage',ascending=False).reset_index(drop=True)


# In[24]:


missing_val


# In[35]:


#lets check what is the best method to impute missing  values
#save output result first
missing_val.to_csv("Missing_perc.csv",index=False)


# In[54]:


#columns which have missing values 
#passenger_count
#fare_amount


# In[48]:


training_data[:30]


# In[25]:


#create some missing values to impute
#orignal value obserbation index 17,,,passenger_count value is 1
training_data['passenger_count'].loc[17]=np.nan


# In[51]:


training_data[:30]


# In[52]:


#impute with mean
training_data['passenger_count']=training_data['passenger_count'].fillna(training_data['passenger_count'].mean())


# In[53]:


training_data[:30]


# In[84]:


#mean method 2.62


# In[57]:


training_data['passenger_count'].loc[17]=np.nan


# In[12]:


#impute with median
training_data['passenger_count']=training_data['passenger_count'].fillna(training_data['passenger_count'].median())


# In[28]:


training_data.isnull().sum()


# In[27]:


training_data[:30]


# In[ ]:


# median method 1.0


# In[38]:


#median method is performing perfectly in this case
training_data['passenger_count']=training_data['passenger_count'].fillna(training_data['passenger_count'].median())


# In[29]:


#lets check for another variable fare_amount
training_data[:50]


# In[30]:


#create some missing value orignal value is 4.50
training_data['fare_amount'].loc[36]=np.nan


# In[31]:


training_data[:50]


# In[8]:


#impute with mean 
training_data['fare_amount']=training_data['fare_amount'].fillna(training_data['fare_amount'].mean())


# In[ ]:


#imputed value with mean is 15.01


# In[25]:


training_data['fare_amount'].loc[36]=np.nan


# In[13]:


training_data['fare_amount']=training_data['fare_amount'].fillna(training_data['fare_amount'].median())


# In[33]:


training_data[:50]


# In[28]:


#median is closest to the orignal so we will impute it with median


# In[37]:


training_data['fare_amount']=training_data['fare_amount'].fillna(training_data['fare_amount'].median())


# In[34]:


training_data[:50]


# In[14]:


training_data.isnull().sum()


# In[16]:


#lets start outlier analysis
#plot boxplot to visualization for outliers
get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(training_data['fare_amount'])


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(training_data['pickup_datetime'])


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(training_data['pickup_longitude'])


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(training_data['dropoff_latitude'])


# In[70]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(training_data['dropoff_longitude'])


# In[15]:



cnames=["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude","passenger_count","pickup_datetime","fare_amount"]


# In[16]:


for i in cnames:
    print(i)
    q75, q25 = np.percentile(training_data.loc[:,i], [75 ,25])
    iqr = q75 - q25

    minimum = q25 - (iqr*1.5)
    maximum = q75 + (iqr*1.5)
    print(minimum)
    print(maximum)
   
    training_data= training_data.drop(training_data[training_data.loc[:,i] < minimum].index)
    training_data = training_data.drop(training_data[training_data.loc[:,i] > maximum].index)


# In[17]:


for i in cnames:
    print(i)
    training_data= training_data.drop(training_data[training_data.loc[:,i] == 0].index)


# In[19]:


training_data.describe()


# In[24]:


ds=training_data.corr()
ds


# In[25]:


f,ag=plt.subplots(figsize=(12,10))
sns.heatmap(ds,mask=np.zeros_like(ds,dtype=np.bool),cmap=sns.diverging_palette(240,30,as_cmap=True),square=True,ax=ag)


# In[26]:


#feature selection 
#corelation analysis
df_corr=training_data.loc[:,cnames]
#set the width and hight of the plot
f,ax=plt.subplots(figsize=(10,8))
#generate corelation matrix
corr=df_corr.corr()
#plot using seaborn library 
sns.heatmap(corr,mask=np.zeros_like(corr,dtype=np.bool),cmap=sns.diverging_palette(220,10,as_cmap=True),square=True,ax=ax)


# In[29]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.pairplot(training_data)


# In[30]:


corrmat = training_data.corr() 
  
f, ax = plt.subplots(figsize =(9, 8)) 
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1)


# In[31]:


cg = sns.clustermap(corrmat, cmap ="YlGnBu", linewidths = 0.1); 
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0) 
  
cg 


# In[44]:


#Normality check
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['passenger_count'],bins='auto')


# In[59]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['fare_amount'],bins='auto')


# In[45]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['pickup_datetime'],bins='auto')


# In[18]:


cnames1=["passenger_count","fare_amount"]


# In[19]:


#Normalization 
for i in cnames1:
    print(i)
    training_data[i]=(training_data[i]-min(training_data[i]))/(max(training_data[i])-min(training_data[i]))


# In[20]:


#pickup_datetime needs standerization 
bc=["pickup_datetime"]
for i in bc:
    print(i)
    training_data[i]=(training_data[i]-training_data[i].mean())/training_data[i].std()


# In[48]:


#chek for Vif score for multicoliniarity 
def vif_cal(input_data,dependent_col):
    import statsmodels.formula.api as smf
    x_vars=input_data.drop([dependent_col],axis=1)
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]]
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=smf.ols(formula="y~x",data=x_vars).fit().rsquared
        vif=round(1/(1-rsq),2)
        print(xvar_names[i],"VIF=",vif)


# In[49]:


vif_cal(input_data=training_data,dependent_col="fare_amount")


# In[18]:


#apply machine learning techniques 
#devide the data into train and test 


# In[21]:


train, test = train_test_split(training_data, test_size=0.2)


# In[22]:


#############Linear Regression #################**************
model=sm.OLS(train.iloc[:,0],train.iloc[:,1:7]).fit()


# In[23]:


model.summary()


# In[24]:


pre=model.predict(test.iloc[:,1:7])


# In[25]:


#calculate MAPE
def MAPE(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100
    return mape


# In[26]:


MAPE(test.iloc[:,0],pre)


# In[ ]:


################Decision Tree Regressor****************


# In[27]:


fit_DT = DecisionTreeRegressor(max_depth=2).fit(train.iloc[:,1:7], train.iloc[:,0])


# In[28]:


predictions_DT = fit_DT.predict(test.iloc[:,1:7])


# In[29]:


MAPE(test.iloc[:,0], predictions_DT)


# In[ ]:


###########KNN for Regression **************


# In[40]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))


# In[41]:


from sklearn.model_selection import train_test_split
train , test = train_test_split(training_data, test_size = 0.3)

x_train = train.iloc[:,1:7]
y_train = train.iloc[:,0]

x_test = test.iloc[:,1:7]
y_test = test.iloc[:,0]


# In[42]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test_scaled)


# In[43]:


#import required packages
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt


# In[44]:


rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model_knn = neighbors.KNeighborsRegressor(n_neighbors = K)

    model_knn.fit(x_train, y_train)  #fit the model
    pred=model_knn.predict(x_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)


# In[45]:


#plotting the rmse values against k values
curve = pd.DataFrame(rmse_val) #elbow curve 
get_ipython().run_line_magic('matplotlib', 'inline')
curve.plot()


# In[46]:


model_knn = neighbors.KNeighborsRegressor(n_neighbors = 6)
model_knn.fit(x_train, y_train)  #fit the model
pred=model_knn.predict(x_test) #make prediction on test set
error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
print('RMSE value for k= ' , "6" , 'is:', error)


# In[47]:


pr1=model_knn.predict(test.iloc[:,1:7])


# In[48]:


MAPE(test.iloc[:,0],pr1)


# In[ ]:


#######********Random Forest Regression*************************


# In[59]:


from sklearn.ensemble import RandomForestRegressor

RF_model = RandomForestRegressor(n_estimators = 20).fit(x_train, y_train)


# In[61]:


RF_Predictions = RF_model.predict(x_test)


# In[62]:


RF_Predictions


# In[ ]:


##callculate MAPE***********


# In[63]:


MAPE(test.iloc[:,0],RF_Predictions)


# In[7]:


test_data.describe()


# In[8]:


test_data.isnull().sum()


# In[49]:


test_data['pickup_datetime'] = pd.to_datetime(test_data['pickup_datetime'],errors='coerce')


# In[50]:


test_data['pickup_datetime']=pd.to_numeric(test_data['pickup_datetime'],errors='coerce')


# In[44]:


test_data['pickup_datetime'].describe()


# In[51]:


t_cnames=["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude","passenger_count","pickup_datetime"]


# In[52]:


#removing outliers from the test data
for i in t_cnames:
     print(i)
     q75, q25 = np.percentile(test_data.loc[:,i], [75 ,25])
     iqr = q75 - q25

     minimum = q25 - (iqr*1.5)
     maximum = q75 + (iqr*1.5)
     print(minimum)
     print(maximum)
    
     test_data= test_data.drop(test_data[test_data.loc[:,i] < minimum].index)
     test_data = test_data.drop(test_data[test_data.loc[:,i] > maximum].index)


# In[53]:


#standerization for datetime
kk=["pickup_datetime"]
for i in kk:
    print(i)
    test_data[i]=(test_data[i]-test_data[i].mean())/test_data[i].std()


# In[ ]:


###make predictions on test data*************


# In[64]:


fare_predictions=RF_model.predict(test_data)


# In[65]:


fare_predictions


# In[66]:


fare_predictions=pd.DataFrame(fare_predictions)


# In[68]:


fare_predictions=fare_predictions.rename(columns={'index':'index',0:'fare_amount'})


# In[ ]:


###save the output file


# In[69]:


fare_predictions.to_csv("fare_pedictions.csv",index=False)


# In[71]:


fare_predictions=pd.read_csv("fare_pedictions.csv",sep=',')


# In[76]:


fare_predictions

