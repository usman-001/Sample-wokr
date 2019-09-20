#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
os.chdir("D:/edvisor data science path/Practice_Python")
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


#load the training dataset 
training_data=pd.read_csv("train.csv",sep=',')


# In[29]:


#load the test dataset 
test_data=pd.read_csv("test.csv",sep=',')


# In[ ]:


#start data Preprocessign 
#as given in the problem statement given dataset have one binary target column,one string column and rest of all are numeric data with some missing values  
#proceeding with the assumptions above now start the data analysis 


# In[5]:


#missing value analysis 
#Creat dataframe with missing percentage 
missing_val=pd.DataFrame(training_data.isnull().sum())

#reset index
missing_val=missing_val.reset_index()

#Rename variable
missing_val=missing_val.rename(columns={'index':'variables',0:'missing_percentage'})


# In[6]:


#calculate percentage
missing_val['missing_percentage']=(missing_val['missing_percentage']/len(training_data))*100


# In[7]:


#sort in decending order 
missing_val=missing_val.sort_values('missing_percentage',ascending=False).reset_index(drop=True)


# In[9]:


#missing value analysis for test data

#Creat dataframe with missing percentage 
missing_val_1=pd.DataFrame(test_data.isnull().sum())

#reset index
missing_val_1=missing_val_1.reset_index()

#Rename variable
missing_val_1=missing_val_1.rename(columns={'index':'variables',0:'missing_percentage'})


# In[10]:


#calculate percentage
missing_val_1['missing_percentage']=(missing_val_1['missing_percentage']/len(test_data))*100


# In[13]:


#sort in decending order 
missing_val_1=missing_val_1.sort_values('missing_percentage',ascending=False).reset_index(drop=True)


# In[11]:


#lets check the training_data datasets what percentage of missing values they have
missing_val


# In[ ]:


#the output above shows that training_data does not have any missing value 


# In[ ]:


#lets check for test dataset 


# In[14]:


missing_val_1


# In[ ]:


#the test data also does not have any missing value 


# In[ ]:


#so we are proceeding further 


# In[15]:


#save output result
missing_val.to_csv("Missing_percentage.csv",index=False)


# In[16]:


missing_val_1.to_csv("Missing_percentage_1.csv",index=False)


# In[8]:


##outlayer Analysis
#plot boxplotto vsualize outliears 
#cheching one by one for every variable 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(training_data['var_0'])


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(training_data['var_1'])


# In[75]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(training_data['var_2'])


# In[76]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(training_data['var_3'])


# In[77]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(training_data['var_4'])


# In[78]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(training_data['var_5'])


# In[79]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(training_data['var_6'])


# In[80]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(training_data['var_89'])


# In[81]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(training_data['var_172'])


# In[82]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(training_data['var_197'])


# In[21]:


#outlayer analysis for test data 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(test_data['var_0'])


# In[22]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(test_data['var_6'])


# In[23]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(test_data['var_197'])


# In[25]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(test_data['var_88'])


# In[ ]:


#from the above interpretation we can say that the data set have out layers 


# In[9]:


#lets detect and delete the outlayers in training_data
#save the numeric variables names firts 

cnames=["var_0","var_1","var_2","var_3","var_4","var_5","var_6","var_7","var_8","var_9","var_11","var_12","var_13","var_14","var_15","var_16","var_17","var_18","var_19","var_20","var_21","var_22","var_23","var_24","var_25","var_26","var_27","var_28","var_29","var_30","var_31","var_32","var_33","var_34","var_35","var_36","var_37","var_38","var_39","var_40","var_41","var_42","var_43","var_44","var_45","var_46","var_47","var_48","var_49","var_50","var_51","var_52","var_53","var_54","var_55","var_56","var_57","var_58","var_59","var_60","var_61","var_62","var_63","var_64","var_65","var_66","var_67","var_68","var_69","var_70","var_71","var_72","var_73","var_74","var_75","var_76","var_77","var_78","var_79","var_80","var_81","var_82","var_83","var_84","var_85","var_86","var_87","var_88","var_89","var_90","var_91","var_92","var_92","var_93","var_94","var_95","var_96","var_97","var_98","var_99","var_100","var_101","var_102","var_103","var_104","var_105","var_106","var_107","var_108","var_109","var_110","var_111","var_112","var_113","var_114","var_115","var_116","var_117","var_118","var_119","var_120","var_121","var_122","var_123","var_124","var_125","var_126","var_127","var_128","var_129","var_130","var_131","var_132","var_133","var_134","var_135","var_136","var_137","var_138","var_139","var_140","var_141","var_142","var_143","var_144","var_145","var_146","var_147","var_148","var_149","var_150","var_151","var_152","var_153","var_154","var_155","var_156","var_157","var_158","var_159","var_160","var_161","var_162","var_163","var_164","var_165","var_166","var_167","var_168","var_169","var_170","var_171","var_172","var_173","var_174","var_175","var_176","var_177","var_178","var_179","var_180","var_181","var_182","var_183","var_184","var_185","var_186","var_187","var_188","var_189","var_190","var_191","var_192","var_193","var_194","var_195","var_196","var_197","var_198","var_199"]


# In[10]:


for i in cnames:
    print(i)
    q75,q25=np.percentile(training_data.loc[:,i],[75,25])
    iqr=q75-q25
    minimum=q25-(iqr*1.5)
    maximum=q75+(iqr*1.5)
    print(minimum)
    print(maximum)
    training_data=training_data.drop(training_data[training_data.loc[:,i]<minimum].index)
    traininig_data=training_data.drop(training_data[training_data.loc[:,i]>maximum].index)


# In[11]:


#feature selection 
df_corr=training_data.loc[:,cnames]


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['target'])


# In[13]:


corr=df_corr.corr()


# In[21]:


a4_dims = (50,30)
fig, ap = plt.subplots(figsize=a4_dims)
sns.set(font_scale=2)
sns.heatmap(corr,mask=np.zeros_like(corr,dtype=np.bool),cmap=sns.diverging_palette(260,10),ax=ap)


# In[24]:


##feature scaling 
#saving the copy of the data
df=training_data.copy()
df1=test_data.copy()


# In[14]:


#normality check
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_0'],bins='auto')


# In[25]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_1'],bins='auto')


# In[91]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_2'],bins='auto')


# In[92]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_3'],bins='auto')


# In[93]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_4'],bins='auto')


# In[28]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_5'],bins='auto')


# In[29]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_6'],bins='auto')


# In[43]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_7'],bins='auto')


# In[44]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_8'],bins='auto')


# In[45]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_9'],bins='auto')


# In[46]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_10'],bins='auto')


# In[47]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_11'],bins='auto')


# In[48]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_12'],bins='auto')


# In[49]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_13'],bins='auto')


# In[50]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_14'],bins='auto')


# In[51]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_15'],bins='auto')


# In[52]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_16'],bins='auto')


# In[53]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_17'],bins='auto')


# In[54]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_18'],bins='auto')


# In[55]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_19'],bins='auto')


# In[56]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_20'],bins='auto')


# In[57]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_21'],bins='auto')


# In[58]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_22'],bins='auto')


# In[59]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_23'],bins='auto')


# In[32]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(training_data['var_199'],bins='auto')


# In[15]:


#apply normalization in training data
for i in cnames:
    print(i)
    training_data[i] = (training_data[i] - min(training_data[i]))/(max(training_data[i]) - min(training_data[i]))


# In[31]:


#apply normalization on test data
for i in cnames:
    print(i)
    test_data[i] = (test_data[i] - min(test_data[i]))/(max(test_data[i]) - min(test_data[i]))


# In[45]:


#now apply different machine learning algorithms to check which is perform better and selelct the best model


# In[16]:


#Replace target categries with yes or no 
training_data['target']=training_data['target'].replace(0,'No')
training_data['target']=training_data['target'].replace(1,'Yes')


# In[17]:


#Divide data into train and test
X = training_data.values[:, 2:203]
Y = training_data.values[:,1]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2)


# In[31]:


#1::Decision Tree
clf = tree.DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train)


# In[32]:


#predict new test cases
Y_predict = clf.predict(X_test)


# In[33]:


Y_predict


# In[34]:


#Error matrix for classification 
CM=confusion_matrix(y_test,Y_predict)


# In[35]:


CM=pd.crosstab(y_test,Y_predict)


# In[36]:


CM


# In[39]:


#let us save TP,TN,FP,FN
TN=CM.iloc[0,0]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]
FP=CM.iloc[0,1]


# In[37]:


#check accuracy of model
accuracy_score(y_test,Y_predict)*100


# In[40]:


#Accuracy
((TP+TN)*100)/(TP+TN+FP+FN)


# In[41]:


#False Negative Rate
(FN*100)/(FN+TP)


# In[42]:


#Recall
(TP*100)/(TP+FN)


# In[ ]:


#lets make a note for DT
#Accuracy=83.71
#FNR=81.07
#Recall=18.92


# In[43]:


#::2:: Random Forest(Classsifier)
RF_model=RandomForestClassifier(n_estimators=100).fit(X_train,y_train)


# In[44]:


#predict some test casses 
RF_predictions=RF_model.predict(X_test)


# In[45]:


CM=confusion_matrix(y_test,RF_predictions)


# In[46]:


CM=pd.crosstab(y_test,RF_predictions)


# In[47]:


CM


# In[48]:


#let us save TP,TN,FP,FN
TN=CM.iloc[0,0]
FN=CM.iloc[1,0]




# In[51]:


#check accuracy of model
accuracy_score(y_test,RF_predictions)*100


# In[52]:


#Accuracy
((TP+TN)*100)/(TP+TN+FP+FN)


# In[53]:


#False Negative Rate
(FN*100)/(FN+TP)


# In[54]:


#Recall
(TP*100)/(TP+FN)


# In[ ]:


#lets make a note for RF
#Accuracy=90.10
#FNR=83.51
#Recall=15.91


# In[55]:


#::3::KNN implementation 
KNN_model=KNeighborsClassifier(n_neighbors=5).fit(X_train,y_train)


# In[56]:


#predict test casses
KNN_predictions=KNN_model.predict(X_test)


# In[57]:


KNN_predictions


# In[58]:


CM=confusion_matrix(y_test,KNN_predictions)


# In[59]:


CM=pd.crosstab(y_test,KNN_predictions)


# In[60]:


CM


# In[61]:


#let us save TP,TN,FP,FN
TN=CM.iloc[0,0]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]
FP=CM.iloc[0,1]


# In[62]:


#check accuracy of model
accuracy_score(y_test,KNN_predictions)*100


# In[63]:


#Accuracy
((TP+TN)*100)/(TP+TN+FP+FN)


# In[64]:


#False Negative Rate
(FN*100)/(FN+TP)


# In[65]:


#Recall
(TP*100)/(TP+FN)


# In[ ]:


#lets make a note for KNN
#Accuracy=90.10
#FNR=99.86
#Recall=0.13


# In[ ]:


#Naive Bays


# In[18]:


Naive_bays_model=GaussianNB().fit(X_train,y_train)


# In[19]:


#Predict test casess
NB_predictions=Naive_bays_model.predict(X_test)


# In[20]:


NB_predictions


# In[21]:


CM=confusion_matrix(y_test,NB_predictions)


# In[22]:


CM=pd.crosstab(y_test,NB_predictions)


# In[23]:


CM


# In[24]:


#let us save TP,TN,FP,FN
TN=CM.iloc[0,0]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]
FP=CM.iloc[0,1]


# In[25]:


#check accuracy of model
accuracy_score(y_test,NB_predictions)*100


# In[26]:


#Accuracy
((TP+TN)*100)/(TP+TN+FP+FN)


# In[27]:


#False Negative Rate
(FN*100)/(FN+TP)


# In[28]:


#Recall
(TP*100)/(TP+FN)


# In[ ]:


#lets make a note for Naive Bays
#Accuracy=92.06
#FNR=64.77
#Recall=35.27


# In[ ]:


#Naive Bayes have the maximum accuracy and lowest false negative rate amoung the above Machine Learning Algorithm so we will use Naive Bayes to predict our data


# In[32]:


testing_data=test_data.values[:, 1:202]


# In[34]:


Predictions=Naive_bays_model.predict(testing_data)


# In[35]:


Predictions


# In[40]:


new_series = pd.Series(Predictions)


# In[41]:


new_series.describe()


# In[ ]:


new_series.to_csv("cust_predictions.csv",index=True)


# In[54]:


cust_predictions=pd.read_csv("cust_predictions.csv",sep=',')


# In[55]:


cust_predictions


# In[65]:


cust_predictions=cust_predictions.rename(columns={'0':'Index Number','No':'predictions'})


# In[66]:


cust_predictions


# In[ ]:





# In[ ]:





# In[67]:





# In[ ]:




