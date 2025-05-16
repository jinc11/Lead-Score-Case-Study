# Lead-Score-Case-Study
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
pd.set_option("display.max_columns",None)
%matplotlib inline

pd.set_option("display.max_columns",None)

df=pd.read_csv("C:\Data Analytics & AI\Lead+Scoring+Case+Study\Lead Scoring Assignment\Leads.csv")

df.shape

df.head()

df.describe()

# It is seen that there are outliers in the 3 variables which are numerical 
# *TotalVisits
# *Total Time Spent on Website
# *Page Views Per Visit
#The same needs to be treated for the analysis

# Looking for the null values in the dataset
round((df.isnull().sum()/df.shape[0])*100,2).sort_values(ascending=False)

#Segregation of Categorical variables and Numerical Variables
Cat_col=df.select_dtypes(include="object")
Num_col=df.select_dtypes(exclude="object")

# Checking for the columns coming under Categorical variables
Cat_col.columns.values

# Checking for the columns coming under Categorical variables
Num_col.columns.values

#Dropping columns that have missing values greater than 40%
Threshold=40
Col_to_keep=df.columns[100*((df.isnull().sum())/df.shape[0])<Threshold]

Col_to_keep.shape

# defining new Dataframe by dropping the unwanted columns
df_n=df[Col_to_keep]

df_n.info()

# SCREENING & WORKING WITH NULL VALUES- CATEGORICAL VARIABLES

round((df_n.isnull().sum()/df_n.shape[0])*100,2).sort_values(ascending=False)

# working with Tags variable null values
(100*(df_n.Tags.value_counts()))/df_n.Tags.shape[0]

# working with Lead Profile variable null values
(100*(df_n["Lead Profile"].value_counts()))/df_n["Lead Profile"].shape[0]

# Tags & Lead Profile variable can be deleted from the analysis as the same is being captured by another variable and 
# it will not be correct to replace with the mode 

# working with What matters most to you in choosing a course variable null values
(100*(df_n["What matters most to you in choosing a course"].value_counts()))/df_n["What matters most to you in choosing a course"].shape[0]

# to fill the null values for "What matters most to you in choosing a course" we will be populating the rows with the value - "Better career prospects" where the Total visits are greater than 3
df_n['What matters most to you in choosing a course'] = np.where(
    (df_n['TotalVisits'] >= 3) & (df_n['What matters most to you in choosing a course'].isna()), 
    'Better Career Prospects', 
    df_n['What matters most to you in choosing a course'])

(100*(df_n["What matters most to you in choosing a course"].value_counts()))/df_n["What matters most to you in choosing a course"].shape[0]

df_n['What matters most to you in choosing a course']=df_n['What matters most to you in choosing a course'].fillna("Other")

# Filling the left out 15% of missing values with others
(100*(df_n["What matters most to you in choosing a course"].value_counts()))/df_n["What matters most to you in choosing a course"].shape[0]

# working with Country variable null values
(100*(df_n.Country.value_counts()))/df_n.Country.shape[0]

# checking the cities data if we can fill the country with india
(100*(df_n.City.value_counts()))/df_n.City.shape[0]

check_country=np.where((df_n['Country'].isna()) & (~df_n['City'].isna()))

check_country

rows_with_na_country = df_n.loc[check_country]
rows_with_na_country.shape

df_n['Country'] = np.where(
    (df_n['City'].isin(["Mumbai", "Thane & Outskirts", "Other Cities of Maharashtra", "Other Metro Cities", "Tier II Cities","Other Cities"])) & 
    (df_n['Country'].isna()), 
    'India', 
    df_n['Country'])


(100*(df_n.Country.value_counts()))/df_n.Country.shape[0]

# Null values treated by using the data from City
100*(df_n.Country.isna().sum()/df_n.Country.shape[0])

# filling the rest of the null values with India
df_n.Country=df_n.Country.fillna("India")

100*(df_n.Country.isna().sum()/df_n.Country.shape[0])

# working with How did you hear about X Education variable null values
(100*(df_n["How did you hear about X Education"].value_counts()))/df_n["How did you hear about X Education"].shape[0]

# Tags & Lead Profile & "How did you hear about X Education" variable can be deleted from the analysis as the same is being captured by another variable and 
# it will not be correct to replace with the mode 

(100*(df_n.Specialization.value_counts()))/df_n.Specialization.shape[0]

# working with City variable null values
(100*(df_n["City"].value_counts()))/df_n["City"].shape[0]

# Tags & Lead Profile & "How did you hear about X Education" & Specialization variable & City can be deleted from the analysis as the same is being captured by another variable and 
# it will not be correct to replace with the mode

# dropping the variables
df_n.drop(columns=["Prospect ID","Tags","Lead Profile","How did you hear about X Education","Specialization","City","What is your current occupation","Country"],inplace=True)

round((df_n.isnull().sum()/df_n.shape[0])*100,2).sort_values(ascending=False)

# Dropping rows that have null values lesser than 0.33 percent
df_n=df_n[~df_n.TotalVisits.isnull()].copy()
df_n=df_n[~df_n["Page Views Per Visit"].isnull()].copy()
df_n=df_n[~df_n["Last Activity"].isnull()].copy()
df_n=df_n[~df_n["Lead Source"].isnull()].copy()

round((df_n.isnull().sum()/df_n.shape[0])*100,2).sort_values(ascending=False)

df_n.info()

# Checking for the numerical variables
Num_col1=df_n.select_dtypes(exclude="object")
Num_col1.columns.values

# checking the value counts for all the object variables 
Cat_col1=df_n.select_dtypes(include="object")
Cat_col1.columns.values

#looking for the variable entries in all these categorcial variables
for c in Cat_col1:
  print("/n The value_count of",c,"is:",df_n[c].value_counts())
  print("*"*80)

#Creating dummy variables for for the categorical variables in the Dataset
df_n1=pd.get_dummies(df_n,columns=['Lead Origin', 'Lead Source', 'Do Not Email',
       'Do Not Call', 'Last Activity',
       'What matters most to you in choosing a course', 'Search',
       'Magazine', 'Newspaper Article', 'X Education Forums', 'Newspaper',
       'Digital Advertisement', 'Through Recommendations',
       'Receive More Updates About Our Courses',
       'Update me on Supply Chain Content', 'Get updates on DM Content',
       'I agree to pay the amount through cheque',
       'A free copy of Mastering The Interview', 'Last Notable Activity'],drop_first=True).astype(int)


df_n1.head()

# NORMALISING THE DATA


# We would be normalising these variables as they are numeric in nature and not considering the dummy variables created
Num_col2=['Lead Number', 'Converted', 'TotalVisits',
       'Total Time Spent on Website', 'Page Views Per Visit']

scaler=MinMaxScaler()
df_n1[Num_col2]=scaler.fit_transform(df_n1[Num_col2])
df_n1.describe()

#plotting heat map to check the variable having the maximum correlation
plt.figure(figsize=(200,100))
sns.heatmap(df_n1.corr(),annot=True)
plt.show()

# SPLITTING THE DATA IN TRAIN & TEST

#splitting the data to test and train for the analysis and prediction and evaluation of the model
df_n1_train,df_n1_test=train_test_split(df_n1,train_size=0.7,random_state=100)
df_n1_train.shape

df_n1_test.shape

# making x train and y train from the test dataset
y_train=df_n1_train.pop("Converted")
X_train=df_n1_train

#Recursive Feature Elimination- selecting the variables that best fit the model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

# creating a logistic regression model
model = LogisticRegression()

n_features_to_select = 10  # Change as needed

rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
rfe.fit(X_train, y_train)

selected_features = X_train.columns[rfe.support_]
print("Selected Features:", selected_features)

feature_ranks=list(zip(X_train.columns,rfe.support_,rfe.ranking_))

sorted_ranks = sorted(feature_ranks, key=lambda x: x[2])
sorted_ranks

top_n = 20
top_features = sorted_ranks[:top_n]
top_features

#Checking the columns taken by the recurive modelling technique 
col=X_train.columns[rfe.support_]
col

#Building the model based on the stats and derived variables 
#i.e. taking the variables selected by rfe for the model
X_train_rfe=X_train[col]
X_train_rfe.head()

#adding constant
X_train_rfe=sm.add_constant(X_train_rfe)

lm = sm.Logit(y_train, X_train_rfe).fit()

print(lm.summary())
