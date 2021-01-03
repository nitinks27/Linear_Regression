#!/usr/bin/env python
# coding: utf-8

# ### About the Data
# 
# I've gathered the dataset of Fuel Consumption, which have specific columns as explained below:
# 
# - **MODELYEAR** e.g. 2014
# - **MAKE** e.g. Acura
# - **MODEL** e.g. ILX
# - **VEHICLE CLASS** e.g. SUV
# - **ENGINE SIZE** e.g. 4.7
# - **CYLINDERS** e.g 6
# - **TRANSMISSION** e.g. A6
# - **FUELTYPE** e.g. z
# - **FUEL CONSUMPTION in CITY(L/100 km)** e.g. 9.9
# - **FUEL CONSUMPTION in HWY (L/100 km)** e.g. 8.9
# - **FUEL CONSUMPTION COMB (L/100 km)** e.g. 9.2
# - **CO2 EMISSIONS (g/km)** e.g. 182   --> low --> 0

# **Task**: `Here the task is to predict the CO2 Emission (dependent variable) from the vechiel in the city Canada, with the help of other essential features (independent variable) available in our dataset.`

# **MODEL SELECTION**: As per the specified task we have to use **Multiple Linear Regression** method to reach our goal.

# ### Regression
# 
# `Regression analysis consists of a set of machine learning methods that allow us to predict a continuous outcome variable (Y) based on the value of one or multiple predictor variables (X).`

# In[1]:


#importing basic libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# **%matplotlib inline** : This is known as magic inline function.
# When using the 'inline' backend, our matplotlib graphs will be included in our notebook, next to the code. 

# Let's have a look at our dataset.

# In[2]:


data = pd.read_csv('Original_2000_2014_Fuel_Consumption_Ratings.csv')


# In[3]:


df = data.copy()
df.head()


# It is a good habit to create copy of our dataframe so that our original data cannot be manipulated during any changes and also it makes us easy to retrive our dataset back, in case of any errors.

# Now, before going forward I've observed some of the columns name have units assigned along with the columns name.
# Let's rename these columns

# In[4]:


df.columns


# In[5]:


df.rename(columns={'ENGINE_SIZE(L)':'ENGINE_SIZE', 'HWY_(L/100km)':'HWY_', 'COMB_(L/100km)':'COMB_', 'COMB_(mpg)':'COMB_mpg', 'FUEL_CONSUMPTION_CITY(L/100km)':'FUEL_CONS_CITY', 'CO2_EMISSIONS(g/km)':'CO2_EMISSIONS'}, inplace=True)


# In[6]:


df.head()


# Perfect!!👌 Let's proceed!

# In[7]:


df.shape


# In[8]:


df.info()


# Here are some basic information about our datasets.

# In[9]:


df.isnull().sum()


# In[10]:


df[df==0].count()


# I've checked for NaN values first and then after go to check for the 0s in our dataset.
# 
# Many of you might think why should I checked for values equals to 0, because sometimes our data is fill with unconditional 0 values inspite of NaN values. So, it must be our habit to check for 0s also while cleaning our dataset.
# 
# `Now firstly let's get a little more description about our continuous columns in our dataset.`

# In[11]:


df.describe()


# Checking our model accuracy in the starting.

# In[12]:


X = df[['MODEL_YEAR','ENGINE_SIZE','CYLINDERS','FUEL_CONS_CITY','HWY_','COMB_','COMB_mpg']]
y = df[['CO2_EMISSIONS']]
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y= train_test_split(X,y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
mlrm = LinearRegression()

mlrm.fit(train_X,train_y)

mlrm.score(train_X, train_y)*100


# It's already a good score! But, let's do whatever we need to do.

# Checking for the visualization of our independent continuous vaiables and the dependent variable.

# In[13]:


plt.figure(figsize=(15,6))
ax = sns.histplot(df['CO2_EMISSIONS'], kde=True)
ax.set_title('KDE HISTPLOT FOR CO2_EMISSIONS', fontsize=14)


# In[14]:


plt.figure(figsize=(15,6))
ax = sns.histplot(df['ENGINE_SIZE'], kde=True)
ax.set_title('KDE HISTPLOT FOR ENGINE_SIZE', fontsize=14)


# In[15]:


plt.figure(figsize=(15,6))
ax = sns.histplot(df['CYLINDERS'], kde=True)
ax.set_title('KDE HISTPLOT FOR CYLINDERS', fontsize=14)


# In[16]:


plt.figure(figsize=(15,6))
ax = sns.histplot(df['FUEL_CONS_CITY'], kde=True)
ax.set_title('KDE HISTPLOT FOR FUEL_CONS_CITY', fontsize=14)


# In[17]:


plt.figure(figsize=(15,6))
ax = sns.histplot(df['HWY_'], kde=True)
ax.set_title('KDE HISTPLOT FOR HWY_', fontsize=14)


# In[18]:


plt.figure(figsize=(15,6))
ax = sns.histplot(df['COMB_'], kde=True)
ax.set_title('KDE HISTPLOT FOR COMB_', fontsize=14)


# In[19]:


plt.figure(figsize=(15,6))
ax = sns.histplot(df['COMB_mpg'], kde=True)
ax.set_title('KDE HISTPLOT FOR COMB_mpg', fontsize=14)


# Pair plot provides an easy and compact visual description of our continuous data.

# In[20]:


ax = sns.pairplot(df)


# Checking for multicollinearity among the continuous columns usinf VIF methods.
# 
# **Multicollinearity**: Multicollinearity occurs when two or more independent variables are highly correlated with one another in a regression model.
# 
# **Why not Multicollinearity?**: Multicollinearity can be a problem in a regression model because we would not be able to distinguish between the individual effects of the independent variables on the dependent variable.
# 
# **Detection of Multicollinearity**: Multicollinearity can be detected via various methods. One of the popular method is using VIF.
# 
# **VIF**: VIF stands for Variable Inflation Factors. VIF determines the strength of the correlation between the independent variables. It is predicted by taking a variable and regressing it against every other variable.
# 
# `Here, I'll check VIF only for the continuous variables.`

# In[21]:


X1= df[['MODEL_YEAR','ENGINE_SIZE','CYLINDERS','FUEL_CONS_CITY','HWY_','COMB_','COMB_mpg']]


# In[22]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X_vif = add_constant(X1)

pd.Series([variance_inflation_factor(X_vif.values, i) 
               for i in range(X_vif.shape[1])], 
              index=X_vif.columns)


# In[ ]:





# Now, VIF of "FUEL_CONS_CITY", "HWY_" and "COMB_" is very very high. That means we have to drop any two of the columns because it's not at all good for our model. 
# 
# But Wait! How will we decide which of the columns should be dropped?
# 
# Here comes the role of Significancy.
# 
# **Significancy**: In statistics, statistical significance means that the result that was produced has a reason behind it, it was not produced randomly, or by chance.
# 
# Here, we are currently focusing on continuous data. So best statistical test according to this condition will be: Correlation Coefficients
# 
# **Correlation Coefficients**: Correlation coefficients are used to measure how strong a relationship is between two variables.

# In[23]:


df.corr()


# In[24]:


plt.figure(figsize=(15,6))
ax = sns.heatmap(df.corr(),annot = True)
ax.set_title('CORRELATION MATRIX', fontsize=14)


# FUEL_CONS_CITY and HWY_ are highly correlated with other independent variables as well as least with CO2_EMISSIONS among three. So, time to say Bye-Bye to these columns one by one to see the VIF chnages.
# 
# Also, MODEL_YEAR is very not in a hood correlation with CO2_EMISSIONS. So, also bye-bye to MODEL_YEAR.

# In[25]:


X_vif = X_vif.drop(['FUEL_CONS_CITY'],axis = 1)
pd.Series([variance_inflation_factor(X_vif.values, i) 
               for i in range(X_vif.shape[1])], 
              index=X_vif.columns)


# In[26]:


X_vif = X_vif.drop(['HWY_'],axis = 1)
pd.Series([variance_inflation_factor(X_vif.values, i) 
               for i in range(X_vif.shape[1])], 
              index=X_vif.columns)


# In[27]:


X_vif = X_vif.drop(['MODEL_YEAR'],axis = 1)
pd.Series([variance_inflation_factor(X_vif.values, i) 
               for i in range(X_vif.shape[1])], 
              index=X_vif.columns)


# Let's see if our model accuracy improved or not after VIF.

# In[28]:


X = df[['ENGINE_SIZE','CYLINDERS','COMB_','COMB_mpg']]
y = df[['CO2_EMISSIONS']]
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y= train_test_split(X,y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
mlrm = LinearRegression()

mlrm.fit(train_X,train_y)

mlrm.score(train_X, train_y)*100


# Not much afftected!! Infact a little less than before. Strange but cool, it is how it is. Let's get into categorical values.

# Now we also have to drop MODEL_YEAR, FUEL_CONS_CITY & HWY_ from our dataset also. So, let's make it done before moving forward.

# In[29]:


df.drop(['MODEL_YEAR','FUEL_CONS_CITY','HWY_'], axis = 1, inplace= True)
df.head()


# In[30]:


df['MAKE'].nunique()


# In[31]:


df['MODEL'].nunique()


# In[32]:


df['VEHICLE_CLASS'].nunique()


# In[33]:


df['TRANSMISSION'].nunique()


# In[34]:


df['FUEL_TYPE'].nunique()


# So, after seeking into catergorical values I find MODEL column has a lot of unique values that might not seem anyway helping us to reach our goal so bye-bye MODEL!!

# In[35]:


df.drop(['MODEL'], axis = 1, inplace= True)


# Countplot gives us the decription about the categories present inside a columns. So, why not makre use of it!

# In[36]:


plt.figure(figsize=(15,6))
ax = sns.countplot(df['MAKE'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title('Count Plot for MAKE', fontsize=14)


# CHEVROLET has the maximum make among all whereas BUGATTI, SRT, ALFA ROMEO PLYMOUTH have a low make figure.

# In[37]:


plt.figure(figsize=(10,6))
ax = sns.countplot(df['VEHICLE_CLASS'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
ax.set_title('Count Plot for VEHICLE_CLASS', fontsize=14)


# SUV winning the customer much more than the others and COMPACT & MOD-SIZE are not much behind the race!!

# In[38]:


plt.figure(figsize=(10,6))
ax = sns.countplot(df['TRANSMISSION'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
ax.set_title('Count Plot for TRANSMISSION', fontsize=14)


# A4 TRANSMISSIONS is much more common trailing by M5, M6, AS6 & A6/

# In[39]:


plt.figure(figsize=(10,6))
ax = sns.countplot(df['FUEL_TYPE'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Count Plot for FUEL_TYPE', fontsize=14)


# X and Z are the preffered FUEL_TYPE by the citizens.

# Comparing categorical variables with dependent variable and visulizing the result.

# In[40]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=df['MAKE'], y=df['CO2_EMISSIONS'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title('MAKE VS CO2_EMISSIONS', fontsize=14)


# These rich brands like BUGATI trailing by LAMBORGHINI, MASERATI, FERRARI have high CO2_EMISSIONS.

# In[41]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=df['VEHICLE_CLASS'], y=df['CO2_EMISSIONS'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
ax.set_title('VEHICLE_CLASS VS CO2_EMISSIONS', fontsize=14)


# VAN types and PICKUP TRUCKS have moderately hight CO2_EMISSIONS.

# In[42]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=df['TRANSMISSION'], y=df['CO2_EMISSIONS'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
ax.set_title('TRANSMISSION VS CO2_EMISSIONS', fontsize=14)


# In[43]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=df['FUEL_TYPE'], y=df['CO2_EMISSIONS'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('FUEL_TYPE VS CO2_EMISSIONS', fontsize=14)


# No doubt why peoplr are preffering X & Z FUEL_TYPE over N.

# I didn't find much visulatization differences between these columns. So, let's move towards the Statistical Checkings!
# 
# For that, firstly we've to encode our data so that our model can understand it perectly.

# **Encoding**: In laymens language it is just converting data into numerical forms so that our model can understand data easily.
# 
# There are many techniques for label encoding but here after observing data well I choose Binary Encoding.
# 
# Many of you might not have installed category_encoders earliers. So it's simple to install with `pip install category_encoders`.

# In[44]:


import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['FUEL_TYPE','MAKE','VEHICLE_CLASS','TRANSMISSION'])
df = encoder.fit_transform(df)
df.head()


# In[45]:


df.shape


# Now after encoding we have 28 columns.

# In[46]:


X1 = df.drop(["CO2_EMISSIONS",'ENGINE_SIZE','CYLINDERS','COMB_','COMB_mpg'],axis = 1)
y = df["CO2_EMISSIONS"]


# In[47]:


import scipy.stats as stats
for i in X1.columns:
    print(stats.f_oneway(X1[i],y))


# None of the pvalue is greater than significance value 0.05. So, it proves there is no relationship between the variables with y, so must not be dropped.

# In[48]:


X = df.drop(["CO2_EMISSIONS"],axis = 1)
y = df["CO2_EMISSIONS"]

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y= train_test_split(X,y, test_size=0.5, random_state=8)

from sklearn.linear_model import LinearRegression
mlrm = LinearRegression()

mlrm.fit(train_X,train_y)

mlrm.score(train_X, train_y)*100


# In[49]:


import statsmodels.api as sm
mod = sm.OLS(train_y, train_X)
res = mod.fit()
print(res.summary())


# In[50]:


plt.figure(figsize=(16,8))
#plt.subplot(211)
plt.plot(test_y.reset_index(drop=True), label='Actual', color='g')
#plt.subplot(212)
plt.plot(mlrm.predict(test_X), label='Predict', color='r')
plt.legend(loc='upper right')


# We can see the predicted line have almost covered the green line very well. Great!!
# 
# Also, score seems pretty good!! Almost 98% values are predicted well. So, Congrats to us!!😊👍
